# ABOUTME: Unit tests for the BridgetOS OpenAI adapter.
# ABOUTME: Covers GovernanceLockedError, BridgetOSAssistantEventHandler, and instrument_openai_agents.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bridgetos_openai import (
    BridgetOSAssistantEventHandler,
    GovernanceLockedError,
    instrument_openai_agents,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_b_client(governance_state: str = "normal", drift_score: float | None = 0.1):
    b_client = MagicMock()
    result = MagicMock()
    result.governance_state = governance_state
    result.drift_score = drift_score
    b_client.observe.return_value = result
    return b_client


def _make_handler(governance_state: str = "normal", **kwargs):
    b_client = _make_b_client(governance_state=governance_state)
    handler = BridgetOSAssistantEventHandler(
        agent_id="test-agent-001",
        b_client=b_client,
        **kwargs,
    )
    return handler, b_client


def _make_function_tool_call(
    name: str = "get_weather",
    arguments: str = '{"city": "NYC"}',
    output: str | None = "sunny",
):
    tc = MagicMock()
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    tc.function.output = output
    return tc


def _make_code_interpreter_tool_call(
    input_text: str = "print(1+1)",
    log_output: str = "2",
):
    tc = MagicMock()
    tc.type = "code_interpreter"
    tc.code_interpreter = MagicMock()
    tc.code_interpreter.input = input_text
    out = MagicMock()
    out.logs = log_output
    tc.code_interpreter.outputs = [out]
    return tc


def _make_run_step(step_type: str = "tool_calls", prompt_tokens: int = 10, completion_tokens: int = 20):
    run_step = MagicMock()
    run_step.type = step_type
    run_step.usage = MagicMock()
    run_step.usage.prompt_tokens = prompt_tokens
    run_step.usage.completion_tokens = completion_tokens
    return run_step


def _make_message(text: str = "Hello, how can I help?"):
    message = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = MagicMock()
    block.text.value = text
    message.content = [block]
    return message


def _make_response(text: str = "Hello", tool_name: str | None = None):
    response = MagicMock()
    response.model = "gpt-4o"
    response.usage = MagicMock()
    response.usage.input_tokens = 10
    response.usage.output_tokens = 20

    output = []
    if text:
        msg_item = MagicMock()
        msg_item.type = "message"
        content_block = MagicMock()
        content_block.type = "output_text"
        content_block.text = text
        msg_item.content = [content_block]
        output.append(msg_item)

    if tool_name:
        fn_item = MagicMock()
        fn_item.type = "function_call"
        fn_item.name = tool_name
        fn_item.arguments = '{"q": "test"}'
        output.append(fn_item)

    response.output = output
    return response


def _make_openai_client(response=None):
    client = MagicMock()
    if response is not None:
        client.responses.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# 1. GovernanceLockedError
# ---------------------------------------------------------------------------

class TestGovernanceLockedError:
    def test_governance_locked_error_message(self):
        err = GovernanceLockedError("agent-123", 0.95)
        assert "agent-123" in str(err)
        assert "0.95" in str(err)

    def test_governance_locked_error_is_runtime_error(self):
        err = GovernanceLockedError("agent-123", None)
        assert isinstance(err, RuntimeError)


# ---------------------------------------------------------------------------
# 2. BridgetOSAssistantEventHandler — tool call accumulation
# ---------------------------------------------------------------------------

class TestToolCallAccumulation:
    def test_function_tool_call_mapped_correctly(self):
        handler, _ = _make_handler()
        tc = _make_function_tool_call(name="get_weather", arguments='{"city": "NYC"}', output="sunny")
        handler.on_tool_call_done(tc)

        assert len(handler._pending_tool_calls) == 1
        mapped = handler._pending_tool_calls[0]
        assert mapped.name == "get_weather"
        assert mapped.arguments == {"city": "NYC"}
        assert mapped.result == "sunny"

    def test_code_interpreter_tool_call_mapped(self):
        handler, _ = _make_handler()
        tc = _make_code_interpreter_tool_call(input_text="x = 2+2", log_output="4")
        handler.on_tool_call_done(tc)

        assert len(handler._pending_tool_calls) == 1
        mapped = handler._pending_tool_calls[0]
        assert mapped.name == "code_interpreter"
        assert mapped.arguments == {"input": "x = 2+2"}
        assert mapped.result == "4"

    def test_multiple_tool_calls_accumulated(self):
        handler, _ = _make_handler()
        handler.on_tool_call_done(_make_function_tool_call(name="tool_a"))
        handler.on_tool_call_done(_make_function_tool_call(name="tool_b"))

        assert len(handler._pending_tool_calls) == 2
        assert handler._pending_tool_calls[0].name == "tool_a"
        assert handler._pending_tool_calls[1].name == "tool_b"

    def test_tool_calls_cleared_after_step_done(self):
        handler, _ = _make_handler()
        handler.on_run_step_created(_make_run_step())
        handler.on_tool_call_done(_make_function_tool_call())
        assert len(handler._pending_tool_calls) == 1

        handler.on_run_step_done(_make_run_step(step_type="tool_calls"))
        assert handler._pending_tool_calls == []


# ---------------------------------------------------------------------------
# 3. BridgetOSAssistantEventHandler — observation emission
# ---------------------------------------------------------------------------

class TestObservationEmission:
    def test_observe_called_on_run_step_done(self):
        handler, b_client = _make_handler()
        handler.on_run_step_created(_make_run_step())
        handler.on_tool_call_done(_make_function_tool_call(name="do_thing"))
        handler.on_run_step_done(_make_run_step(step_type="tool_calls"))

        b_client.observe.assert_called_once()
        obs = b_client.observe.call_args[0][0]
        assert obs.agent_id == "test-agent-001"
        assert obs.content.text == ""
        assert len(obs.content.tool_calls) == 1
        assert obs.content.tool_calls[0].name == "do_thing"

    def test_observe_called_on_message_done(self):
        handler, b_client = _make_handler()
        handler.on_run_step_created(_make_run_step())
        handler.on_message_done(_make_message("Hello world"))

        b_client.observe.assert_called_once()
        obs = b_client.observe.call_args[0][0]
        assert obs.content.text == "Hello world"
        assert obs.content.tool_calls == []

    def test_observe_not_called_on_message_creation_step(self):
        handler, b_client = _make_handler()
        handler.on_run_step_created(_make_run_step())
        # message_creation type run step must not emit from on_run_step_done
        handler.on_run_step_done(_make_run_step(step_type="message_creation"))

        b_client.observe.assert_not_called()


# ---------------------------------------------------------------------------
# 4. BridgetOSAssistantEventHandler — governance
# ---------------------------------------------------------------------------

class TestAssistantHandlerGovernance:
    def test_governance_locked_raises_exception(self):
        handler, _ = _make_handler(governance_state="locked")
        handler.on_run_step_created(_make_run_step())
        handler.on_tool_call_done(_make_function_tool_call())

        with pytest.raises(GovernanceLockedError):
            handler.on_run_step_done(_make_run_step(step_type="tool_calls"))

    def test_governance_normal_does_not_raise(self):
        handler, _ = _make_handler(governance_state="normal")
        handler.on_run_step_created(_make_run_step())
        handler.on_tool_call_done(_make_function_tool_call())
        handler.on_run_step_done(_make_run_step(step_type="tool_calls"))  # no exception

    def test_log_errors_true_swallows_api_error(self):
        handler, b_client = _make_handler(log_errors=True)
        b_client.observe.side_effect = Exception("connection refused")
        handler.on_run_step_created(_make_run_step())
        handler.on_tool_call_done(_make_function_tool_call())
        handler.on_run_step_done(_make_run_step(step_type="tool_calls"))  # must not raise

    def test_log_errors_false_propagates_api_error(self):
        handler, b_client = _make_handler(log_errors=False)
        b_client.observe.side_effect = Exception("connection refused")
        handler.on_run_step_created(_make_run_step())
        handler.on_tool_call_done(_make_function_tool_call())

        with pytest.raises(Exception, match="connection refused"):
            handler.on_run_step_done(_make_run_step(step_type="tool_calls"))


# ---------------------------------------------------------------------------
# 5. instrument_openai_agents — patching and observation emission
# ---------------------------------------------------------------------------

class TestInstrumentOpenaiAgents:
    def test_responses_create_patched(self):
        openai_client = _make_openai_client(_make_response())
        original = openai_client.responses.create
        instrument_openai_agents(openai_client, MagicMock(), agent_id="a1")
        assert openai_client.responses.create is not original

    def test_responses_observe_called_on_create(self):
        b_client = _make_b_client()
        response = _make_response(text="The answer is 42")
        openai_client = _make_openai_client(response)
        instrument_openai_agents(openai_client, b_client, agent_id="a1", session_id="s1")
        openai_client.responses.create(input="What is 6x7?")

        b_client.observe.assert_called_once()
        obs = b_client.observe.call_args[0][0]
        assert obs.agent_id == "a1"
        assert obs.session_id == "s1"
        assert obs.content.text == "The answer is 42"

    def test_responses_function_tool_call_mapped(self):
        b_client = _make_b_client()
        response = _make_response(text="", tool_name="search_web")
        openai_client = _make_openai_client(response)
        instrument_openai_agents(openai_client, b_client, agent_id="a1")
        openai_client.responses.create(input="search something")

        b_client.observe.assert_called_once()
        obs = b_client.observe.call_args[0][0]
        assert len(obs.content.tool_calls) == 1
        assert obs.content.tool_calls[0].name == "search_web"

    def test_responses_no_observe_on_empty_output(self):
        b_client = _make_b_client()
        response = _make_response(text="", tool_name=None)  # output=[]
        openai_client = _make_openai_client(response)
        instrument_openai_agents(openai_client, b_client, agent_id="a1")
        openai_client.responses.create(input="test")

        b_client.observe.assert_not_called()

    def test_responses_governance_locked_raises(self):
        b_client = _make_b_client(governance_state="locked")
        response = _make_response(text="hello")
        openai_client = _make_openai_client(response)
        instrument_openai_agents(openai_client, b_client, agent_id="a1", halt_on_lock=True)

        with pytest.raises(GovernanceLockedError):
            openai_client.responses.create(input="test")

    def test_original_response_returned_unchanged(self):
        b_client = _make_b_client()
        response = _make_response(text="result")
        openai_client = _make_openai_client(response)
        instrument_openai_agents(openai_client, b_client, agent_id="a1")
        returned = openai_client.responses.create(input="test")

        assert returned is response
