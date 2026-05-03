# Contributing to bridgetos-openai

OpenAI Assistants and Responses API adapter. Published as `bridgetos-openai` on PyPI.

## Setup

```bash
cd packages/bridgetos-openai
uv sync --extra dev
```

## Test commands

```bash
uv run pytest
uv run pytest --cov --cov-report=term-missing
```

## Lint and format

```bash
uv run ruff check src/
uv run ruff format src/
```

CI runs both as `--check` (no auto-fix). Fix locally before pushing.

## Adapter architecture

This package covers two distinct OpenAI SDK surfaces, each with a different interception strategy.

### Assistants API — `BridgetOSAssistantEventHandler`

Subclasses `openai.AssistantEventHandler`, which exposes 17 typed `on_*` callbacks for the deprecated Assistants streaming path. The handler follows the same accumulate-then-flush pattern as `bridgetos-langchain`:

1. **Accumulate** — `on_tool_call_done` appends each completed tool call to `_pending_tool_calls`.
2. **Flush (tool step)** — `on_run_step_done` emits one `Observation` when `run_step.type == "tool_calls"`, bundling all accumulated tool calls, then clears the buffer.
3. **Flush (message step)** — `on_message_done` emits one `Observation` with the assistant's text output and an empty tool calls list.

This produces **one `Observation` per run step**, not one per tool call or per token.

### Responses API — `instrument_openai_agents`

Monkey-patches `openai_client.responses.create` on a specific client instance. Each call to `responses.create` emits one `Observation` after the call returns, extracting text from `output[*].type == "message"` items and tool calls from `output[*].type == "function_call"` items. If both are empty, no observation is emitted.

### Governance

Both paths call `Client.observe()` and check the result. If `governance_state == "locked"` and `halt_on_lock=True`, `GovernanceLockedError` is raised. Callers must propagate this to halt agent execution.

## Adding new event hooks (Assistants path)

1. Identify which `AssistantEventHandler` callback fires at the right lifecycle point.
2. If it's a step-level event: accumulate into `_pending_tool_calls` or a similar buffer.
3. If it's a completion event: flush the buffer and call `Client.observe()`.
4. Add a test that verifies the resulting `Observation` contains the expected content.

## Adding new output item types (Responses path)

1. Identify the `item.type` string for the new output type (e.g. `"web_search_call"`).
2. Add an extraction branch in `_emit_response_observation`.
3. Map it to a `ToolCall` with appropriate `name`, `arguments`, and `result` fields.
4. Add a test with a mocked `Response` containing that output item type.

## Testing against real APIs

Set credentials and point at a local or staging API:

```bash
export BRIDGETOS_API_KEY=your-key
export BRIDGETOS_BASE_URL=https://api.bridgetos.com
export OPENAI_API_KEY=your-openai-key
```

For unit tests, patch `Client.observe` so no real HTTP calls are made.
