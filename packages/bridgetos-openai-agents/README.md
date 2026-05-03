# bridgetos-openai-agents

BridgetOS adapter for the [openai-agents](https://github.com/openai/openai-agents-python) SDK.
Hooks into native tracing to emit behavioral observations to BridgetOS and halt execution on governance lock.

Licensed under the [MIT License](../../LICENSE).

## Installation

```bash
pip install bridgetos-openai-agents
```

## Usage

```python
from bridgetos import Client
from bridgetos_openai_agents import instrument_openai_agents
from agents import Agent, Runner

client = Client(api_key="YOUR_BRIDGETOS_API_KEY")
instrument_openai_agents(client, agent_id="my-agent")

agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
result = await Runner.run(agent, "What is the capital of France?")
print(result.final_output)
```

If BridgetOS locks the agent's governance state, `BridgetOSGovernanceException` is raised automatically, halting execution.
