# ABOUTME: Public API for bridgetos-openai-agents — exports processor, exception, and setup helper.
# ABOUTME: Call instrument_openai_agents(client) once at startup to wire in the BridgetOS trace processor.

from __future__ import annotations

from bridgetos import Client

from .processor import BridgetOSGovernanceException, BridgetOSTraceProcessor

__all__ = [
    "BridgetOSTraceProcessor",
    "BridgetOSGovernanceException",
    "instrument_openai_agents",
]


def instrument_openai_agents(
    bridgetos_client: Client, agent_id: str = "openai-agents"
) -> None:
    """Register the BridgetOS trace processor with the openai-agents SDK."""
    from agents.tracing import add_trace_processor

    processor = BridgetOSTraceProcessor(bridgetos_client, agent_id=agent_id)
    add_trace_processor(processor)
