# ABOUTME: OpenAI Assistants and Responses API adapter that emits BridgetOS observations.
# ABOUTME: Provides BridgetOSAssistantEventHandler (native hook subclass) and instrument_openai_agents() (Responses wrapper).

from __future__ import annotations

import functools
import json
import logging
import time
from typing import Any

from openai import AssistantEventHandler

from bridgetos import Client, Observation, ObservationContent, ObservationContext, ObservationTelemetry
from bridgetos import ToolCall as BridgetOSToolCall

logger = logging.getLogger(__name__)


class GovernanceLockedError(RuntimeError):
    """Raised when BridgetOS has locked the agent's governance state."""

    def __init__(self, agent_id: str, drift_score: float | None) -> None:
        super().__init__(
            f"BridgetOS has locked agent {agent_id!r} (drift_score={drift_score}). "
            "Execution halted at governance layer."
        )
        self.agent_id = agent_id
        self.drift_score = drift_score


class BridgetOSAssistantEventHandler(AssistantEventHandler):
    """OpenAI Assistants streaming event handler that emits BridgetOS observations.

    Args:
        agent_id: Stable identifier for the agent.
        b_client: Pre-configured BridgetOS Client.
        session_id: Optional session identifier to group related observations.
        task_type: Optional task type label (e.g., 'customer_support').
        model: Optional model name override; falls back to the run's model.
        halt_on_lock: If True (default), raise GovernanceLockedError on locked state.
        log_errors: If True (default), log API errors instead of propagating them.
    """

    def __init__(
        self,
        agent_id: str,
        b_client: Client,
        session_id: str | None = None,
        task_type: str | None = None,
        model: str | None = None,
        halt_on_lock: bool = True,
        log_errors: bool = True,
    ) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.b_client = b_client
        self.session_id = session_id
        self.task_type = task_type
        self.model = model
        self.halt_on_lock = halt_on_lock
        self.log_errors = log_errors
        self._pending_tool_calls: list[BridgetOSToolCall] = []
        self._run_start: float = 0.0

    # AssistantEventHandler hooks --------------------------------------------

    def on_run_step_created(self, run_step: Any) -> None:
        self._run_start = time.monotonic()

    def on_tool_call_done(self, tool_call: Any) -> None:
        self._pending_tool_calls.append(self._map_tool_call(tool_call))

    def on_run_step_done(self, run_step: Any) -> None:
        if run_step.type != "tool_calls":
            return
        elapsed_ms = (time.monotonic() - self._run_start) * 1000
        usage = run_step.usage
        observation = Observation(
            agent_id=self.agent_id,
            session_id=self.session_id,
            content=ObservationContent(
                text="",
                tool_calls=self._pending_tool_calls,
            ),
            context=ObservationContext(
                model=self.model,
                framework="openai-assistants",
                task_type=self.task_type,
            ),
            telemetry=ObservationTelemetry(
                latency_ms=elapsed_ms,
                tokens_input=usage.prompt_tokens if usage else None,
                tokens_output=usage.completion_tokens if usage else None,
            ),
        )
        self._pending_tool_calls = []
        self._submit(observation)

    def on_message_done(self, message: Any) -> None:
        text = ""
        for block in message.content:
            if block.type == "text":
                text = block.text.value
                break
        elapsed_ms = (time.monotonic() - self._run_start) * 1000
        observation = Observation(
            agent_id=self.agent_id,
            session_id=self.session_id,
            content=ObservationContent(
                text=text,
                tool_calls=[],
            ),
            context=ObservationContext(
                model=self.model,
                framework="openai-assistants",
                task_type=self.task_type,
            ),
            telemetry=ObservationTelemetry(
                latency_ms=elapsed_ms,
            ),
        )
        self._submit(observation)

    def on_exception(self, exception: Exception) -> None:
        logger.warning("OpenAI stream exception: %s", exception)

    # Internals --------------------------------------------------------------

    def _map_tool_call(self, tool_call: Any) -> BridgetOSToolCall:
        tc_type = tool_call.type
        if tc_type == "function":
            try:
                args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            return BridgetOSToolCall(
                name=tool_call.function.name,
                arguments=args,
                result=tool_call.function.output,
            )
        elif tc_type == "code_interpreter":
            input_text = tool_call.code_interpreter.input or ""
            outputs = tool_call.code_interpreter.outputs or []
            result = None
            for out in outputs:
                if hasattr(out, "logs"):
                    result = out.logs
                    break
            return BridgetOSToolCall(
                name="code_interpreter",
                arguments={"input": input_text},
                result=result,
            )
        else:
            # file_search or unrecognised type
            fs = getattr(tool_call, "file_search", None)
            result_count = None
            if fs and hasattr(fs, "results") and fs.results:
                result_count = str(len(fs.results))
            return BridgetOSToolCall(
                name="file_search",
                arguments={},
                result=result_count,
            )

    def _submit(self, observation: Observation) -> None:
        try:
            result = self.b_client.observe(observation)
        except Exception as exc:
            if self.log_errors:
                logger.warning("BridgetOS observe failed: %s", exc)
                return
            raise
        if self.halt_on_lock and result.governance_state == "locked":
            raise GovernanceLockedError(self.agent_id, result.drift_score)


def instrument_openai_agents(
    openai_client: Any,
    b_client: Client,
    *,
    agent_id: str,
    session_id: str | None = None,
    task_type: str | None = None,
    model: str | None = None,
    halt_on_lock: bool = True,
    log_errors: bool = True,
) -> None:
    """Monkey-patch openai_client.responses.create to emit BridgetOS observations.

    After calling this, every invocation of openai_client.responses.create() will
    transparently emit an Observation to BridgetOS and raise GovernanceLockedError
    if the agent's governance state is locked.
    """
    original_create = openai_client.responses.create

    @functools.wraps(original_create)
    def _wrapped_create(*args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        response = original_create(*args, **kwargs)
        elapsed_ms = (time.monotonic() - t0) * 1000
        _emit_response_observation(
            response,
            b_client,
            agent_id,
            elapsed_ms,
            session_id=session_id,
            task_type=task_type,
            model=model,
            halt_on_lock=halt_on_lock,
            log_errors=log_errors,
        )
        return response

    openai_client.responses.create = _wrapped_create


def _emit_response_observation(
    response: Any,
    b_client: Client,
    agent_id: str,
    elapsed_ms: float,
    *,
    session_id: str | None,
    task_type: str | None,
    model: str | None,
    halt_on_lock: bool,
    log_errors: bool,
) -> None:
    """Extract text and tool calls from a Response object and submit to BridgetOS."""
    text = ""
    tool_calls: list[BridgetOSToolCall] = []

    for item in (response.output or []):
        if item.type == "message":
            for block in (item.content or []):
                if block.type == "output_text":
                    text = block.text
                    break
        elif item.type == "function_call":
            try:
                args = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(BridgetOSToolCall(name=item.name, arguments=args))

    if not text and not tool_calls:
        return

    usage = response.usage
    resolved_model = model or (getattr(response, "model", None))
    observation = Observation(
        agent_id=agent_id,
        session_id=session_id,
        content=ObservationContent(text=text, tool_calls=tool_calls),
        context=ObservationContext(
            model=resolved_model,
            framework="openai-responses",
            task_type=task_type,
        ),
        telemetry=ObservationTelemetry(
            latency_ms=elapsed_ms,
            tokens_input=usage.input_tokens if usage else None,
            tokens_output=usage.output_tokens if usage else None,
        ),
    )

    try:
        result = b_client.observe(observation)
    except Exception as exc:
        if log_errors:
            logger.warning("BridgetOS observe failed: %s", exc)
            return
        raise

    if halt_on_lock and result.governance_state == "locked":
        raise GovernanceLockedError(agent_id, result.drift_score)
