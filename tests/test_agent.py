"""Offline tests for the agent loop. A scripted model means no network/API key.

Tests are async; run them with `python run_tests.py` (no pytest needed) or with
pytest if you have pytest-asyncio.
"""

from drangue import Agent, ModelResponse, ToolCall, tool
from drangue.models import Model


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class ScriptedModel(Model):
    """Returns a pre-baked sequence of responses, one per generate call."""

    def __init__(self, steps):
        self.steps = steps
        self.i = 0
        self.calls = 0

    async def generate(self, *, system, messages, tools):
        self.calls += 1
        resp = self.steps[self.i]
        self.i += 1
        return resp


async def test_tool_loop_runs_tool_then_answers():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("t1", "add", {"a": 2, "b": 3})]),
        ModelResponse(text="The answer is 5."),
    ])
    agent = Agent(model=model, tools=[add])
    result = await agent.run("what is 2 + 3?")

    assert result.output == "The answer is 5."
    assert result.steps == 2


def test_schema_is_generated_from_signature():
    schema = add.to_schema()
    assert schema["name"] == "add"
    props = schema["input_schema"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "integer"
    assert set(schema["input_schema"]["required"]) == {"a", "b"}


async def test_unknown_tool_is_reported_not_raised():
    bad = ToolCall("x", "nope", {})
    model = ScriptedModel([
        ModelResponse(tool_calls=[bad]),
        ModelResponse(text="done"),
    ])
    agent = Agent(model=model, tools=[add])
    result = await agent.run("go")
    assert result.output == "done"


async def test_stream_yields_events():
    call = ToolCall("t1", "add", {"a": 1, "b": 1})
    model = ScriptedModel([
        ModelResponse(tool_calls=[call]),
        ModelResponse(text="two"),
    ])
    agent = Agent(model=model, tools=[add])
    types = [e.type async for e in agent.stream("go")]
    assert types == [
        "run_started", "model_decision", "tool_result",
        "model_decision", "run_finished",
    ]


async def test_async_tool_is_awaited():
    async def fetch(city: str) -> str:
        """Fetch something asynchronously."""
        return f"data for {city}"

    call = ToolCall("t1", "fetch", {"city": "Paris"})
    model = ScriptedModel([
        ModelResponse(tool_calls=[call]),
        ModelResponse(text="done"),
    ])
    agent = Agent(model=model, tools=[fetch])
    result = await agent.run("go")
    assert result.output == "done"
    # The tool result is recorded in the log.
    tool_results = [e for e in result.events if e.type == "tool_result"]
    assert tool_results[0].payload["content"] == "data for Paris"


def test_plain_function_is_accepted_as_tool():
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(model=ScriptedModel([]), tools=[multiply])
    assert "multiply" in agent.tools
