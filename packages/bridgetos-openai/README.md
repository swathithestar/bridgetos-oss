# bridgetos-openai

OpenAI Assistants and Responses API integration for BridgetOS. Automatically emits observations from OpenAI agent runs and halts execution if BridgetOS revokes the agent's credential.

## Install

```bash
pip install bridgetos-openai
```

## Usage — Assistants API (streaming)

```python
import openai
from bridgetos import Client
from bridgetos_openai import BridgetOSAssistantEventHandler

b_client = Client(api_key="b_os_...")
handler = BridgetOSAssistantEventHandler(agent_id="support-bot-001", b_client=b_client)

client = openai.OpenAI()
with client.beta.threads.runs.stream(
    thread_id=thread.id,
    assistant_id=assistant.id,
    event_handler=handler,
) as stream:
    stream.until_done()
```

## Usage — Responses API

```python
import openai
from bridgetos import Client
from bridgetos_openai import instrument_openai_agents

b_client = Client(api_key="b_os_...")
client = openai.OpenAI()

instrument_openai_agents(client, b_client, agent_id="support-bot-001")

# All subsequent calls are automatically traced and governed
response = client.responses.create(model="gpt-4o", input="What is the weather?")
```

Each agent step is submitted to BridgetOS as a single observation. If BridgetOS reports `governance_state == 'locked'`, the adapter raises `GovernanceLockedError` to halt execution.

## License

MIT
