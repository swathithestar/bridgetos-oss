 Plan: bridgetos-openai-agents — Phase 2 Implementation

 Context

 The Phase 1 research (thoughts/shared/research/2026-05-02-openai-sdk-execution-lifecycle.md) identified two distinct observability surfaces in the openai Python SDK (v2.33.0) for building the bridgetos-openai-agents adapter:

 - Assistants API (deprecated): 17 native on_* callbacks via subclassable AssistantEventHandler → Option A
 - Responses API (modern preferred): pure SSE iterator with no hooks → Option B (wrap/monkey-patch)

 This plan implements both as a single package. Work happens on a new feature branch, not main.

 ---
 Reference Files

 - Sibling adapter pattern: packages/bridgetos-langchain/src/bridgetos_langchain/callback.py
 - SDK models: packages/bridgetos-sdk-python/src/bridgetos/schema.py
 - SDK client: packages/bridgetos-sdk-python/src/bridgetos/client.py
 - Observation schema: packages/bridgetos-schema/observation-v1.json
 - Research: thoughts/shared/research/2026-05-02-openai-sdk-execution-lifecycle.md

 ---
 Package to Create

 packages/bridgetos-openai-agents/
 ├── pyproject.toml
 ├── README.md
 ├── LICENSE
 ├── src/
 │   └── bridgetos_openai_agents/
 │       ├── __init__.py
 │       └── instrumentation.py
 └── tests/
     └── test_instrumentation.py

 ---
 Implementation Details

 pyproject.toml

 Mirror bridgetos-langchain/pyproject.toml exactly:
 - name = "bridgetos-openai-agents", version = "0.1.0"
 - build-backend = "hatchling.build", packages = ["src/bridgetos_openai_agents"]
 - dependencies = ["bridgetos>=0.1.0", "openai>=1.0.0"]
 - [project.optional-dependencies] dev = ["pytest>=7.0.0"]

 instrumentation.py — Three Public Exports

 1. GovernanceLockedError(RuntimeError)

 Same shape as bridgetos_langchain.GovernanceLockedError:
 def __init__(self, agent_id: str, drift_score: float | None) -> None

 2. BridgetOSAssistantEventHandler(AssistantEventHandler)

 Subclass of openai.lib.streaming._assistants.AssistantEventHandler (the native hook surface).

 State:
 - _pending_tool_calls: list[ToolCall] — accumulates during on_tool_call_done
 - _run_start: float — latency tracking via time.monotonic()

 Hook implementations:

 ┌───────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
 │             Hook              │                                      Action                                       │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ on_run_step_created(run_step) │ Record start time                                                                 │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ on_tool_call_done(tool_call)  │ Append to _pending_tool_calls                                                     │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ on_run_step_done(run_step)    │ If type == "tool_calls": emit Observation with accumulated tool calls, clear list │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ on_message_done(message)      │ Emit Observation with message text, empty tool_calls                              │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ on_exception(exc)             │ Log warning (don't re-raise)                                                      │
 └───────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘

 Data mapping for on_run_step_done (tool_calls step):
 Observation(
     agent_id=self.agent_id,
     session_id=self.session_id,
     content=ObservationContent(
         text="",  # tool-only step has no text output
         tool_calls=self._pending_tool_calls,
     ),
     context=ObservationContext(model=self.model, framework="openai-assistants", task_type=self.task_type),
     telemetry=ObservationTelemetry(
         latency_ms=elapsed_ms,
         tokens_input=run_step.usage.prompt_tokens if run_step.usage else None,
         tokens_output=run_step.usage.completion_tokens if run_step.usage else None,
     ),
 )

 Data mapping for on_message_done (message step):
 - Extract text from message.content[i] where content[i].type == "text" → content[i].text.value
 - tool_calls=[] (message steps carry no tool calls)

 Tool call mapping (in on_tool_call_done):

 ┌────────────────────┬─────────────────────────┬─────────────────────────────────────────────┬───────────────────────────┐
 │   ToolCall type    │      ToolCall.name      │             ToolCall.arguments              │      ToolCall.result      │
 ├────────────────────┼─────────────────────────┼─────────────────────────────────────────────┼───────────────────────────┤
 │ "function"         │ tool_call.function.name │ json.loads(tool_call.function.arguments)    │ tool_call.function.output │
 ├────────────────────┼─────────────────────────┼─────────────────────────────────────────────┼───────────────────────────┤
 │ "code_interpreter" │ "code_interpreter"      │ {"input": tool_call.code_interpreter.input} │ first output logs string  │
 ├────────────────────┼─────────────────────────┼─────────────────────────────────────────────┼───────────────────────────┤
 │ "file_search"      │ "file_search"           │ {}                                          │ result count as string    │
 └────────────────────┴─────────────────────────┴─────────────────────────────────────────────┴───────────────────────────┘

 Constructor:
 def __init__(
     self,
     agent_id: str,
     b_client: Client,
     session_id: str | None = None,
     task_type: str | None = None,
     model: str | None = None,
     halt_on_lock: bool = True,
     log_errors: bool = True,
 ) -> None

 3. instrument_openai_agents(openai_client, b_client, *, agent_id, session_id=None, task_type=None, model=None, halt_on_lock=True, log_errors=True)

 Monkey-patches the responses resource on the provided openai.OpenAI client instance.

 What it patches: openai_client.responses.create

 How:
 original_create = openai_client.responses.create

 @functools.wraps(original_create)
 def _wrapped_create(*args, **kwargs):
     t0 = time.monotonic()
     response = original_create(*args, **kwargs)  # call original
     elapsed_ms = (time.monotonic() - t0) * 1000
     _emit_response_observation(response, b_client, agent_id, elapsed_ms, ...)
     return response  # always return unchanged

 openai_client.responses.create = _wrapped_create

 _emit_response_observation mapping:
 - Text: response.output[i] where item.type == "message" → item.content[j].text (first OutputText block)
 - Tool calls: response.output[i] where item.type == "function_call" → ToolCall(name=item.name, arguments=json.loads(item.arguments))
 - Tokens: response.usage.input_tokens, response.usage.output_tokens
 - If text is empty and no tool calls: skip emit (nothing meaningful to observe)

 Governance: After _emit_response_observation calls b_client.observe(), if result.governance_state == "locked" and halt_on_lock=True, raise GovernanceLockedError.

 init.py exports

 from bridgetos_openai_agents.instrumentation import (
     BridgetOSAssistantEventHandler,
     GovernanceLockedError,
     instrument_openai_agents,
 )
 __version__ = "0.1.0"
 __all__ = ["BridgetOSAssistantEventHandler", "GovernanceLockedError", "instrument_openai_agents"]

 ---
 Test Plan (TDD — tests written before implementation)

 All tests in tests/test_instrumentation.py. Use unittest.mock only; no real OpenAI or BridgetOS network calls.

 Test groups

 1. GovernanceLockedError
 - test_governance_locked_error_message — str contains agent_id and drift_score
 - test_governance_locked_error_is_runtime_error — isinstance check

 2. BridgetOSAssistantEventHandler — tool call accumulation
 - test_function_tool_call_mapped_correctly — FunctionToolCall → ToolCall with name/args
 - test_code_interpreter_tool_call_mapped — CodeInterpreterToolCall → ToolCall
 - test_multiple_tool_calls_accumulated — two on_tool_call_done calls → two ToolCalls in observation
 - test_tool_calls_cleared_after_step_done — _pending_tool_calls resets after on_run_step_done

 3. BridgetOSAssistantEventHandler — observation emission
 - test_observe_called_on_run_step_done — mock b_client.observe; verify called with correct Observation
 - test_observe_called_on_message_done — message text extracted correctly
 - test_observe_not_called_on_message_creation_step — message_creation run step type does NOT trigger emission on on_run_step_done (only on_message_done does)

 4. BridgetOSAssistantEventHandler — governance
 - test_governance_locked_raises_exception — mock b_client.observe returns ObservationResult(governance_state="locked") → GovernanceLockedError raised
 - test_governance_normal_does_not_raise — governance_state="normal" → no exception
 - test_log_errors_true_swallows_api_error — b_client.observe raises Exception → no re-raise when log_errors=True
 - test_log_errors_false_propagates_api_error — b_client.observe raises Exception → re-raises when log_errors=False

 5. instrument_openai_agents — patching
 - test_responses_create_patched — after instrument call, mock openai_client.responses.create is wrapped
 - test_responses_observe_called_on_create — calling wrapped create → b_client.observe called with correct Observation
 - test_responses_function_tool_call_mapped — ResponseFunctionToolCall in output → ToolCall in observation
 - test_responses_no_observe_on_empty_output — response with no message/tool output → observe NOT called
 - test_responses_governance_locked_raises — locked response → GovernanceLockedError
 - test_original_response_returned_unchanged — wrapped create still returns the original Response object

 ---
 Implementation Steps

 Step 0: Branch

 git checkout -b feat/bridgetos-openai-agents

 Step 1: Scaffold package skeleton

 Create all files with stubs (enough to import cleanly):
 - pyproject.toml, LICENSE (copy from langchain), README.md (stub)
 - src/bridgetos_openai_agents/__init__.py (empty exports)
 - src/bridgetos_openai_agents/instrumentation.py (class/function stubs, all methods raise NotImplementedError)

 Step 2: Write all tests (failing)

 Write tests/test_instrumentation.py in full. Run uv run pytest → all tests fail.

 Step 3: Implement GovernanceLockedError

 Tests for #1 pass.

 Step 4: Implement BridgetOSAssistantEventHandler

 Implement tool call accumulation, mapping, and emission. Tests for #2, #3, #4 pass.

 Step 5: Implement instrument_openai_agents

 Implement Responses API wrapping. Tests for #5 pass.

 Step 6: Full test suite green

 uv run pytest -v — all tests pass, no errors or warnings.

 Step 7: Finalize README

 5-line wire-in example for each path.

 Step 8: uv run ruff check src/ — clean

 ---
 Verification

 cd packages/bridgetos-openai-agents
 uv sync --extra dev
 uv run pytest -v          # all tests pass
 uv run ruff check src/    # no lint errors
 uv build                  # wheel builds cleanly

 Confirm:
 - GovernanceLockedError is raised when governance_state == "locked" (test covers this)
 - Observation schema valid: each emitted Observation has non-empty agent_id, valid content.text (can be ""), and correct tool_calls shape
 - instrument_openai_agents returns the original Response object unchanged (transparent to caller)