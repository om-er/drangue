"""Offline tests. They use a scripted model, so no network or API key needed."""

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
        self.seen_tools = None

    def generate(self, *, system, messages, tools):
        self.seen_tools = tools
        resp = self.steps[self.i]
        self.i += 1
        return resp


def _tool_use_msg(call):
    return {"role": "assistant", "content": [
        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
    ]}


def _text_msg(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def test_tool_loop_runs_tool_then_answers():
    call = ToolCall(id="t1", name="add", arguments={"a": 2, "b": 3})
    model = ScriptedModel([
        ModelResponse(tool_calls=[call], assistant_message=_tool_use_msg(call)),
        ModelResponse(text="The answer is 5.",
                      assistant_message=_text_msg("The answer is 5.")),
    ])

    agent = Agent(model=model, tools=[add])
    result = agent.run("what is 2 + 3?")

    assert result.output == "The answer is 5."
    assert result.steps == 2


def test_schema_is_generated_from_signature():
    schema = add.to_schema()
    assert schema["name"] == "add"
    props = schema["input_schema"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "integer"
    assert set(schema["input_schema"]["required"]) == {"a", "b"}


def test_unknown_tool_is_reported_not_raised():
    bad = ToolCall(id="x", name="nope", arguments={})
    model = ScriptedModel([
        ModelResponse(tool_calls=[bad], assistant_message=_tool_use_msg(bad)),
        ModelResponse(text="done", assistant_message=_text_msg("done")),
    ])
    agent = Agent(model=model, tools=[add])
    result = agent.run("go")
    assert result.output == "done"


def test_stream_yields_events():
    call = ToolCall(id="t1", name="add", arguments={"a": 1, "b": 1})
    model = ScriptedModel([
        ModelResponse(tool_calls=[call], assistant_message=_tool_use_msg(call)),
        ModelResponse(text="two", assistant_message=_text_msg("two")),
    ])
    agent = Agent(model=model, tools=[add])
    types = [e.type for e in agent.stream("go")]
    assert types == ["model", "tool_call", "tool_result", "model", "final"]


def test_plain_function_is_accepted_as_tool():
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(model=ScriptedModel([]), tools=[multiply])
    assert "multiply" in agent.tools
