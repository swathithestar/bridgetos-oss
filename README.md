# BridgetOS Open Source

Open-source integration layer for BridgetOS — the runtime behavioral identity platform for autonomous AI agents.

## What's here

| Package | Purpose | License |
|---|---|---|
| `bridgetos-schema` | JSON Schema for the observation data contract | MIT |
| `bridgetos-sdk-python` | Python client for sending observations | MIT |
| `bridgetos-langchain` | LangChain callback that emits BridgetOS observations | MIT |
| `bridgetos-openai-agents` | openai-agents SDK adapter that emits BridgetOS observations | MIT |
| `bridgetos-test-harness` | Synthetic agents + drift scenarios for stress-testing identity monitors | Apache 2.0 |

## What's not here

The BridgetOS analytic core — the scoring algorithms, vector weights, calibration parameters, threshold logic, governance state machine implementation, identity credential issuance, and the integrated runtime — is proprietary and runs as a hosted service.

The high-level architecture (Horizon-Defined Identity Tensor, multi-vector behavioral monitoring, governance state classification, derivative detection) is described in our patent filings: US Patent App. 19/651,602 and HDIT Provisional 64/002,373.

This repository contains only the integration surface: the schema agents emit, the client that sends them, framework adapters, and test infrastructure. None of it implements the analytic core; it calls a remote BridgetOS API.

## Quick start

```bash
pip install bridgetos
```

```python
from bridgetos import Client, Observation

client = Client(api_key="your-key", base_url="https://api.bridgetos.com")

obs = Observation(
    agent_id="my-agent",
    content={"text": "The agent's output"},
    context={"task_type": "customer_support"},
)
client.observe(obs)
```

## License

MIT for SDK, schema, and adapters. Apache 2.0 for the test harness.

The BridgetOS analytic core is proprietary. This repository's contents do not grant any rights to the underlying patent claims; they are an integration layer that calls a remote BridgetOS API.

## Contributing

Issues and PRs welcome on individual packages. See each package's `CONTRIBUTING.md`.
