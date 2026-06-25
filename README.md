# drangue

A tiny, obvious agent runtime for Python.

An agent is just a model plus tools. Running it is one call. No graphs, no
chains, no base classes to inherit. You can read the whole loop in one sitting.

## Install

```bash
pip install drangue
```

Set your key:

```bash
export ANTHROPIC_API_KEY=sk-...
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

result = agent.run("What should I wear in Paris today?")
print(result.output)
```

That is the entire surface for the happy path. A tool is a typed function.
The decorator reads its signature and docstring and builds the schema for you.
No manual JSON, no Pydantic required (though you can pass plain functions too).

## See every step

Inspection is one flag, not a separate service:

```python
result = agent.run("What should I wear in Paris today?", trace=True)
```

```
* model
* tool   get_weather(city='Paris')
       -> Paris: 72F and sunny
* model  It is 72F and sunny in Paris, so light layers are perfect.
```

## Drive the loop yourself

`stream` yields an event per step, so you stay in control:

```python
for event in agent.stream("What is the weather in Tokyo?"):
    if event.type == "tool_call":
        print("calling", event.name, event.arguments)
    elif event.type == "final":
        print(event.text)
```

## Bring your own model

`model` can be a string (the default Anthropic adapter) or any object with a
`generate` method. That seam is how you swap providers, add caching, or pass a
fake model in tests. See `tests/test_agent.py` for a fully offline example.

```python
from drangue import AnthropicModel

agent = Agent(
    model=AnthropicModel("claude-opus-4-8", max_tokens=8192),
    tools=[get_weather],
)
```

## What drangue does not do

Keeping these out is the point. It is what protects the obvious feeling.

- No graph or DAG concept. It is a loop.
- No prompt-template engine. f-strings are fine.
- No built-in RAG or vector store. That is a different library.
- No provider-specific code in the core. Adapters are swappable.
- No mandatory config files.

## Roadmap

- Durability: resume a run exactly where it stopped after a crash.
- Human in the loop: pause, wait for approval or input, resume.
- Async API alongside the sync one.
- More model adapters.

The same event stream that powers `trace` today is what durability will be
built on. The basic API will not change.

## Develop

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
