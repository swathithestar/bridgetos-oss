# ABOUTME: Unit tests for BridgetOSTraceProcessor — generation/function span mapping and governance lock.
# ABOUTME: Uses MagicMock to simulate openai-agents span objects without a live API.

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from agents import FunctionSpanData, GenerationSpanData

from bridgetos import Observation
from bridgetos.schema import ObservationResult
from bridgetos_openai_agents.processor import BridgetOSGovernanceException, BridgetOSTraceProcessor


def _make_span(span_data: object, started_at: str | None = None, ended_at: str | None = None) -> MagicMock:
    span = MagicMock()
    span.span_data = span_data
    span.span_id = "span-001"
    span.trace_id = "trace-001"
    span.started_at = started_at
    span.ended_at = ended_at
    return span


def _make_result(**kwargs) -> ObservationResult:
    defaults = {"observation_id": "obs-001", "governance_state": None}
    defaults.update(kwargs)
    return ObservationResult(**defaults)


class TestGenerationSpan:
    def test_calls_observe_with_correct_fields(self):
        client = MagicMock()
        client.observe.return_value = _make_result()

        span_data = GenerationSpanData(
            input=[{"role": "user", "content": "hello"}],
            output=[{"role": "assistant", "content": "hi there"}],
            model="gpt-4o",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        span = _make_span(span_data)

        processor = BridgetOSTraceProcessor(client, agent_id="test-agent")
        processor.on_span_end(span)

        client.observe.assert_called_once()
        obs: Observation = client.observe.call_args[0][0]
        assert obs.agent_id == "test-agent"
        assert obs.context.model == "gpt-4o"
        assert obs.context.framework == "openai-agents"
        assert obs.telemetry.tokens_input == 10
        assert obs.telemetry.tokens_output == 5
        assert "hi there" in obs.content.text

    def test_output_fallback_to_json_when_no_content_field(self):
        client = MagicMock()
        client.observe.return_value = _make_result()

        span_data = GenerationSpanData(
            output=[{"role": "assistant", "custom": "value"}],
            model="gpt-4o",
        )
        span = _make_span(span_data)

        processor = BridgetOSTraceProcessor(client)
        processor.on_span_end(span)

        obs: Observation = client.observe.call_args[0][0]
        assert obs.content.text  # some non-empty fallback

    def test_prompt_token_alias_supported(self):
        """prompt_tokens / completion_tokens are accepted as aliases."""
        client = MagicMock()
        client.observe.return_value = _make_result()

        span_data = GenerationSpanData(
            output=[{"role": "assistant", "content": "ok"}],
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 7, "completion_tokens": 3},
        )
        span = _make_span(span_data)

        processor = BridgetOSTraceProcessor(client)
        processor.on_span_end(span)

        obs: Observation = client.observe.call_args[0][0]
        assert obs.telemetry.tokens_input == 7
        assert obs.telemetry.tokens_output == 3


class TestFunctionSpan:
    def test_calls_observe_with_tool_call(self):
        client = MagicMock()
        client.observe.return_value = _make_result()

        span_data = FunctionSpanData(name="search_web", input='{"query": "test"}', output="results")
        span = _make_span(span_data)

        processor = BridgetOSTraceProcessor(client, agent_id="tool-agent")
        processor.on_span_end(span)

        client.observe.assert_called_once()
        obs: Observation = client.observe.call_args[0][0]
        assert obs.agent_id == "tool-agent"
        assert obs.context.framework == "openai-agents"
        assert len(obs.content.tool_calls) == 1
        assert obs.content.tool_calls[0].name == "search_web"
        assert obs.content.tool_calls[0].result == "results"

    def test_tool_call_with_no_output(self):
        client = MagicMock()
        client.observe.return_value = _make_result()

        span_data = FunctionSpanData(name="notify", input=None, output=None)
        span = _make_span(span_data)

        processor = BridgetOSTraceProcessor(client)
        processor.on_span_end(span)

        obs: Observation = client.observe.call_args[0][0]
        assert obs.content.tool_calls[0].result is None


class TestGovernanceLock:
    def test_raises_exception_when_locked(self):
        client = MagicMock()
        client.observe.return_value = _make_result(governance_state="locked")

        span_data = GenerationSpanData(
            output=[{"role": "assistant", "content": "do bad thing"}],
            model="gpt-4o",
        )
        span = _make_span(span_data)

        processor = BridgetOSTraceProcessor(client)
        with pytest.raises(BridgetOSGovernanceException):
            processor.on_span_end(span)

    def test_does_not_raise_when_not_locked(self):
        client = MagicMock()
        client.observe.return_value = _make_result(governance_state="normal")

        span_data = GenerationSpanData(
            output=[{"role": "assistant", "content": "ok"}],
            model="gpt-4o",
        )
        span = _make_span(span_data)

        processor = BridgetOSTraceProcessor(client)
        processor.on_span_end(span)  # should not raise


class TestNonMatchingSpan:
    def test_non_generation_non_function_span_ignored(self):
        from agents import AgentSpanData

        client = MagicMock()
        span_data = AgentSpanData(name="MyAgent")
        span = _make_span(span_data)

        processor = BridgetOSTraceProcessor(client)
        processor.on_span_end(span)

        client.observe.assert_not_called()


class TestLatencyCalculation:
    def test_latency_calculated_from_span_timestamps(self):
        client = MagicMock()
        client.observe.return_value = _make_result()

        span_data = GenerationSpanData(
            output=[{"role": "assistant", "content": "latency test"}],
            model="gpt-4o",
        )
        span = _make_span(
            span_data,
            started_at="2025-01-01T00:00:00.000Z",
            ended_at="2025-01-01T00:00:01.500Z",
        )

        processor = BridgetOSTraceProcessor(client)
        processor.on_span_end(span)

        obs: Observation = client.observe.call_args[0][0]
        assert obs.telemetry.latency_ms == pytest.approx(1500.0, abs=1.0)

    def test_latency_none_when_timestamps_missing(self):
        client = MagicMock()
        client.observe.return_value = _make_result()

        span_data = GenerationSpanData(
            output=[{"role": "assistant", "content": "no time"}],
            model="gpt-4o",
        )
        span = _make_span(span_data, started_at=None, ended_at=None)

        processor = BridgetOSTraceProcessor(client)
        processor.on_span_end(span)

        obs: Observation = client.observe.call_args[0][0]
        assert obs.telemetry is None or obs.telemetry.latency_ms is None
