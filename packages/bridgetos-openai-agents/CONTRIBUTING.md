# Contributing to bridgetos-openai-agents

openai-agents SDK `TracingProcessor` adapter. Published as `bridgetos-openai-agents` on PyPI.

## Setup

```bash
cd packages/bridgetos-openai-agents
uv sync --extra dev
```

## Test commands

```bash
uv run pytest
uv run pytest --cov --cov-report=term-missing
```

## Lint and format

```bash
uv run ruff check src/
uv run ruff format src/
```

CI runs both as `--check` (no auto-fix). Fix locally before pushing.

## Processor architecture

`BridgetOSTraceProcessor` hooks into the openai-agents SDK's native tracing via `TracingProcessor` — no monkey-patching. It dispatches on span type using `isinstance` checks against `GenerationSpanData` and `FunctionSpanData`:

- **Generation spans** — LLM completions. Extracts output text, model name, token usage, and latency, then emits a single `Observation` via `Client.observe()`.
- **Function spans** — Tool/function calls. Emits an `Observation` with a `ToolCall` in `content.tool_calls`.
- **All other span types** — silently ignored; `observe()` is not called.

Each span produces **at most one `Observation`**. There is no accumulate-then-flush pattern here — spans arrive individually and are submitted immediately on `on_span_end`.

If `observe()` returns `governance_state == "locked"`, the processor raises `BridgetOSGovernanceException`. This propagates out of the openai-agents run loop, halting execution.

## Adding support for new span types

1. Identify the span data class (e.g. `ResponseSpanData`, `HandoffSpanData`) and its fields in `agents/tracing/span_data.py`.
2. Add an `isinstance` branch in `on_span_end`.
3. Add a private `_handle_<type>` method that builds an `Observation` and calls `_submit`.
4. Write a test that verifies the resulting `Observation` fields and that non-matching spans are not submitted.

## Testing against a real openai-agents run

Set credentials and point at a local or staging API:

```bash
export BRIDGETOS_API_KEY=your-key
export BRIDGETOS_BASE_URL=https://api.bridgetos.com
export OPENAI_API_KEY=your-openai-key
```

Then call `instrument_openai_agents(client)` before `Runner.run(...)` and verify observations appear in the API.

For unit tests, pass a `MagicMock` client and construct span objects directly — no real HTTP calls needed.
