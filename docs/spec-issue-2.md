# Development Specification: `bridgetos-openai-agents`

## Objective
Create a drop-in adapter for the official OpenAI Python Agents SDK that automatically captures agent steps, formats them as BridgetOS Observations, and transmits them to the BridgetOS backend, halting execution if governance locks are triggered.

---

## Phase 1: Research & Discovery (Code Search)

Before writing any implementation code, you must execute a code search against the `openai` Python SDK repository or your local virtual environment to understand its execution lifecycle. 

### Task 1.1: Identify the Agent Execution Entry Point
* **Search Target:** `openai` Python SDK codebase.
* **Keywords:** `class Agent`, `def run(`, `beta.assistants`, `Swarm`.
* **Goal:** Determine the exact class and method users invoke to run an agent loop (e.g., `client.beta.assistants.create` or the newer experimental Agents abstractions).

### Task 1.2: Hunt for Native Observability Hooks
* **Search Target:** The execution method identified in Task 1.1.
* **Keywords:** `Event`, `Callback`, `on_message`, `on_tool_call`, `on_step`, `Listener`.
* **Goal:** Determine if OpenAI provides native hooks. 
    * *If Yes:* We will build a subclass of their Event Listener.
    * *If No:* We will need to monkey-patch/wrap the agent execution loop (similar to the Langfuse implementation).

### Task 1.3: Analyze the Output Payload Schema
* **Search Target:** The yield/return types of the agent execution.
* **Keywords:** `RunStep`, `Message`, `ToolCall`, `FunctionCall`.
* **Goal:** Map out exactly what data structures the OpenAI SDK returns on each step, as this will dictate how we transform the data into the BridgetOS schema.

---

## Phase 2: Implementation Architecture

Based on the findings from Phase 1, you will implement the adapter. This phase assumes the standard structure required by the `bridgetos-oss` monorepo.

### Task 2.1: Package Scaffolding
Create the new package matching the sibling `bridgetos-langchain` package:
* **Path:** `packages/bridgetos-openai-agents/`
* **Files needed:**
    * `pyproject.toml` (Dependencies: `bridgetos-sdk-python`, `openai`)
    * `src/bridgetos_openai_agents/__init__.py`
    * `src/bridgetos_openai_agents/instrumentation.py` (or `callbacks.py`)
    * `tests/test_instrumentation.py`
    * `README.md`
    * `LICENSE` (MIT)

### Task 2.2: The Interceptor (Extraction)
Implement the mechanism to catch the agent's internal steps.

**Option A: Native Callback (If found in Phase 1)**
```python
# Pseudo-code expectation
class BridgetOSAgentEventHandler(OpenAIAgentEventHandler):
    def __init__(self, bridgetos_client):
        self.b_client = bridgetos_client

    def on_step_completed(self, step_data):
        self._process_and_observe(step_data)
```

**Option B: SDK Wrapper (If no native callbacks exist)**
```python
# Pseudo-code expectation
def instrument_openai_agents(bridgetos_client):
    original_run = openai.agents.run # (or whatever the target method is)
    
    @wraps(original_run)
    def wrapped_run(*args, **kwargs):
        response_stream = original_run(*args, **kwargs)
        for chunk in response_stream:
            _process_and_observe(chunk, bridgetos_client)
            yield chunk
            
    openai.agents.run = wrapped_run
```

### Task 2.3: The Transformer (Data Mapping)
Map the OpenAI SDK's specific output to the standard BridgetOS schema.
* **Input:** OpenAI `RunStep`, `Message`, or `ToolCall` objects.
* **Process:** * Extract `input_prompt`, `agent_id`, `tool_used`, `tool_arguments`, and `completion`.
    * Calculate usage/tokens if available in the payload.
* **Output:** A valid `Observation` dictionary matching `bridgetos-schema/observation-v1.json`.

### Task 2.4: The Transmitter & Governance (Action)
Send the transformed observation to BridgetOS and enforce governance.
* Invoke `bridgetos_client.observe(observation_data)`.
* Inspect the response from `Client.observe()`.
* **Critical Requirement:** If the response payload contains `{"governance_state": "locked"}`, you must immediately raise a custom exception (e.g., `BridgetOSGovernanceException("Agent execution halted by BridgetOS governance lock.")`) to stop the agent from proceeding to the next step.

---

## Phase 3: Testing & Documentation

### Task 3.1: Mocked Unit Tests
* Create `tests/test_adapter.py`.
* Use `unittest.mock` to mock the OpenAI SDK responses (preventing actual network calls to OpenAI).
* Mock `bridgetos_sdk_python.Client.observe()` to verify it receives the correctly formatted JSON schema.
* Write a specific test asserting that a mocked `locked` response successfully raises the exception and breaks the agent loop.

### Task 3.2: README & Wire-in Example
Write a concise `README.md` with a 5-line example of how a developer adds this to their code. 

```python
import openai
from bridgetos.client import Client
from bridgetos_openai_agents import instrument_openai_agents

# 1. Initialize BridgetOS
b_client = Client(api_key="b_os_...")

# 2. Wire in the adapter
instrument_openai_agents(b_client)

# 3. Run the official OpenAI agent normally
agent = openai.agents.create(...)
agent.run("What is the weather?") # Automatically traced & governed!
```