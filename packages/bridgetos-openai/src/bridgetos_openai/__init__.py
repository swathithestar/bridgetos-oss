# ABOUTME: Public API for the bridgetos-openai adapter package.
# ABOUTME: Re-exports the three primary symbols used to wire BridgetOS into OpenAI agent runs.

from bridgetos_openai.instrumentation import (
    BridgetOSAssistantEventHandler,
    GovernanceLockedError,
    instrument_openai_agents,
)

__version__ = "0.1.0"
__all__ = ["BridgetOSAssistantEventHandler", "GovernanceLockedError", "instrument_openai_agents"]
