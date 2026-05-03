# ABOUTME: BridgetOS TracingProcessor for the openai-agents SDK.
# ABOUTME: Intercepts generation and function spans, emits BridgetOS observations, and enforces governance locks.

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from agents import FunctionSpanData, GenerationSpanData, TracingProcessor
from agents.tracing import Trace

from bridgetos import (
    Client,
    Observation,
    ObservationContent,
    ObservationContext,
    ObservationTelemetry,
    ToolCall,
)

logger = logging.getLogger(__name__)

_ISO_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
)


class BridgetOSGovernanceException(RuntimeError):
    """Raised when BridgetOS has locked the agent's governance state."""

    def __init__(
        self, message: str = "Agent execution halted by BridgetOS governance lock."
    ) -> None:
        super().__init__(message)


class BridgetOSTraceProcessor(TracingProcessor):
    """openai-agents TracingProcessor that emits each span to BridgetOS for behavioral monitoring."""

    def __init__(self, client: Client, agent_id: str = "openai-agents") -> None:
        self.bridgetos_client = client
        self.agent_id = agent_id

    # TracingProcessor interface -----------------------------------------------

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        pass

    def on_span_start(self, span: Any) -> None:
        pass

    def on_span_end(self, span: Any) -> None:
        data = span.span_data
        if isinstance(data, GenerationSpanData):
            self._handle_generation(span)
        elif isinstance(data, FunctionSpanData):
            self._handle_function(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    # Span handlers ------------------------------------------------------------

    def _handle_generation(self, span: Any) -> None:
        data: GenerationSpanData = span.span_data
        usage = data.usage or {}

        tokens_input = usage.get("input_tokens") or usage.get("prompt_tokens")
        tokens_output = usage.get("output_tokens") or usage.get("completion_tokens")
        latency = self._latency_ms(span)

        text = self._extract_generation_text(data.output)

        observation = Observation(
            agent_id=self.agent_id,
            content=ObservationContent(text=text),
            context=ObservationContext(
                model=data.model,
                framework="openai-agents",
            ),
            telemetry=ObservationTelemetry(
                latency_ms=latency,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
            )
            if (
                latency is not None
                or tokens_input is not None
                or tokens_output is not None
            )
            else None,
        )
        self._submit(observation)

    def _handle_function(self, span: Any) -> None:
        data: FunctionSpanData = span.span_data
        output_str = str(data.output) if data.output is not None else None
        text = (
            f"[Tool: {data.name}] {output_str}"
            if output_str is not None
            else f"[Tool: {data.name}]"
        )

        observation = Observation(
            agent_id=self.agent_id,
            content=ObservationContent(
                text=text,
                tool_calls=[ToolCall(name=data.name, result=output_str)],
            ),
            context=ObservationContext(framework="openai-agents"),
        )
        self._submit(observation)

    def _submit(self, observation: Observation) -> None:
        result = self.bridgetos_client.observe(observation)
        if result.governance_state == "locked":
            raise BridgetOSGovernanceException()

    # Helpers ------------------------------------------------------------------

    def _latency_ms(self, span: Any) -> float | None:
        started = span.started_at
        ended = span.ended_at
        if not started or not ended:
            return None
        try:
            t_start = self._parse_iso(started)
            t_end = self._parse_iso(ended)
            if t_start is None or t_end is None:
                return None
            return (t_end - t_start).total_seconds() * 1000
        except Exception:
            return None

    @staticmethod
    def _parse_iso(ts: str) -> datetime | None:
        for fmt in _ISO_FORMATS:
            try:
                dt = datetime.strptime(ts, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return None

    @staticmethod
    def _extract_generation_text(output: Any) -> str:
        if not output:
            return ""
        texts = []
        for msg in output:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
        if texts:
            return " ".join(texts)
        return json.dumps(output)
