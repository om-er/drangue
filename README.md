# drangue

A tiny, obvious agent runtime for Python.

An agent is just a model plus tools. Running it is one call. No graphs, no
chains, no base classes to inherit. You can read the whole loop in one sitting.

## Install

The core has zero dependencies. Install the adapter for the backend you want:

```bash
pip install "drangue[openai]"     # OpenAI, DeepSeek, Groq, Ollama, and more
pip install "drangue[anthropic]"  # Claude
```

## The whole thing

```python
from drangue import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 72F and sunny"

agent = Agent(
    model="claude-opus-4-8",
    tools=[get_weather],
    instructions="You are a helpful weather assistant.",
)

result = agent.run_sync("What should I wear in Paris today?")
print(result.output)
```

That is the entire surface for the happy path. A tool is a typed function.
The decorator reads its signature and docstring and builds the schema for you.
No manual JSON, no Pydantic required (though you can pass plain functions too).

The core is async. `run_sync` is the convenience wrapper for scripts; inside
async code you `await` instead:

```python
result = await agent.run("What should I wear in Paris today?")
```

## See every step

Inspection is one flag, not a separate service:

```python
result = agent.run_sync("What should I wear in Paris today?", trace=True)
```

```
* tool   get_weather(city='Paris')
       -> Paris: 72F and sunny
* model  It is 72F and sunny in Paris, so light layers are perfect.
```

`result.usage` reports the token totals for the run, and `result.events` is the
full event log the run was driven from.

## Drive the loop yourself

`stream` yields each event as it is appended to the log, so you stay in control:

```python
async for event in agent.stream("What is the weather in Tokyo?"):
    if event.type == "model_decision":
        for call in event.payload["tool_calls"]:
            print("calling", call["name"], call["arguments"])
    elif event.type == "run_finished":
        print(event.payload["output"])
```

## Cheap and local models

`drangue` ships two adapters. One of them, `OpenAIModel`, talks to any
OpenAI-compatible endpoint, which is most of the cheap and free backends. You
choose the backend with `base_url`; the agent loop does not change.

```python
from drangue import Agent, OpenAIModel

# Free and local. Install Ollama, run `ollama pull llama3.1`. No API key, no per-token cost.
agent = Agent(
    model=OpenAIModel("llama3.1", base_url="http://localhost:11434/v1", api_key="ollama"),
    tools=[get_weather],
)
```

Swap the model line for a cheap hosted backend without touching anything else:

| Backend | How |
| --- | --- |
| Ollama / LM Studio | `OpenAIModel("llama3.1", base_url="http://localhost:11434/v1", api_key="ollama")` (free, local) |
| DeepSeek | `OpenAIModel("deepseek-chat", base_url="https://api.deepseek.com")` |
| Groq | `OpenAIModel("llama-3.1-8b-instant", base_url="https://api.groq.com/openai/v1")` |
| OpenRouter | `OpenAIModel("...", base_url="https://openrouter.ai/api/v1")` |
| OpenAI | `OpenAIModel("gpt-4o-mini")` |
| Claude | `"claude-opus-4-8"` or `AnthropicModel("claude-opus-4-8")` |

`api_key` and `base_url` fall back to the `OPENAI_API_KEY` and
`OPENAI_BASE_URL` environment variables when omitted. See `examples/cheap.py`.

## Bring your own model

`model` can be a string (the default Anthropic adapter), one of the adapters
above, or any object with an async `generate` method. That seam is how you swap
providers, add caching, or pass a fake model in tests. The OpenAI and Anthropic
adapters are both tested fully offline against fake clients, see
`tests/test_openai_model.py`.

## What drangue does not do

Keeping the surface obvious is the point.

- No graph or DAG concept. You write a normal agent; the runtime drives the loop.
- No prompt-template engine. f-strings are fine.
- No built-in RAG or vector store. That is a different library.
- No provider-specific code in the core. Adapters are swappable.
- No mandatory config files.

## Architecture

The simple facade sits on a small, durable-by-design core: an **orchestrator**
decides each step deterministically, an **executor** performs it, and every step
is appended to a **store** as an event log. The log is the source of truth; the
run is a fold of it. That shape is what lets observability, durability, and
recovery layer on without changing the facade. See `ROADMAP.md`.

## Roadmap

The current focus is the production core (`ROADMAP.md`):

- Done: orchestrator/executor split, event log, async core, token capture.
- Next: persistent traces (spans, cost, timing), then durable resume after a
  crash, then hardened tool calls (timeouts, retries, validation).

## Develop

```bash
pip install -e ".[dev]"
python run_tests.py     # no pytest needed; uses a tiny async runner
```

## License

MIT
