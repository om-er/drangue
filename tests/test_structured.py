"""Structured output: schema-constrained final answers.

The model delivers via a synthetic final_answer tool (or plain JSON text);
the engine validates, records, and never dispatches it. All offline.
"""

import json

from drangue import Agent, InMemoryStore, ModelResponse, ToolCall, tool
from drangue.models import Model
from drangue.structured import validate

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
        "confidence": {"type": "string", "enum": ["low", "high"]},
    },
    "required": ["answer"],
}


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = steps
        self.i = 0
        self.seen_tools = []

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        self.seen_tools.append([t.name for t in tools])
        resp = self.steps[self.i]
        self.i += 1
        return resp


async def test_final_answer_call_is_validated_recorded_and_never_dispatched():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "final_answer",
                                           {"answer": 5, "confidence": "high"})]),
    ])
    result = await Agent(model=model, tools=[add]).run(
        "what is 2+3?", output_schema=SCHEMA)

    assert result.output_parsed == {"answer": 5, "confidence": "high"}
    assert json.loads(result.output) == {"answer": 5, "confidence": "high"}
    assert result.status == "completed"
    # The model saw the synthetic tool alongside the real one.
    assert "final_answer" in model.seen_tools[0] and "add" in model.seen_tools[0]


async def test_invalid_final_answer_gets_a_clean_retry():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "final_answer",
                                           {"answer": "five"})]),   # wrong type
        ModelResponse(tool_calls=[ToolCall("c2", "final_answer",
                                           {"answer": 5})]),
    ])
    result = await Agent(model=model, tools=[]).run("2+3?", output_schema=SCHEMA)

    assert result.output_parsed == {"answer": 5}
    failures = [e for e in result.events if e.type == "tool_result"
                and not json.loads(e.payload["content"]).get("ok", True)]
    assert len(failures) == 1
    assert json.loads(failures[0].payload["content"])["error"]["category"] == "validation"


async def test_plain_json_text_finish_is_accepted():
    model = ScriptedModel([ModelResponse(text='{"answer": 7}')])
    result = await Agent(model=model, tools=[]).run("3+4?", output_schema=SCHEMA)
    assert result.output_parsed == {"answer": 7}


async def test_nonconforming_text_gets_feedback_then_gives_up_honestly():
    model = ScriptedModel([
        ModelResponse(text="the answer is five"),      # not JSON
        ModelResponse(text="still just prose"),        # correction 1 ignored
        ModelResponse(text="nope"),                    # correction 2 ignored
    ])
    result = await Agent(model=model, tools=[]).run("2+3?", output_schema=SCHEMA)

    assert result.status == "completed"
    assert result.output == "nope"                     # honest unstructured finish
    assert result.output_parsed is None
    feedback = [e for e in result.events if e.type == "schema_feedback"]
    assert len(feedback) == 2
    assert "did not match" in feedback[0].payload["text"]


async def test_structured_run_replays_and_uses_real_tools_first():
    store = InMemoryStore()
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "add", {"a": 2, "b": 3})]),
        ModelResponse(tool_calls=[ToolCall("c2", "final_answer", {"answer": 5})]),
    ])
    agent = Agent(model=model, tools=[add], store=store)
    first = await agent.run("2+3?", run_id="s1", output_schema=SCHEMA)
    assert first.output_parsed == {"answer": 5}

    # Replay: same run_id, no new model calls, schema comes from the log.
    again = await agent.run("2+3?", run_id="s1")
    assert again.output_parsed == {"answer": 5}
    assert model.i == 2


def test_validator_covers_the_documented_subset():
    schema = {"type": "object", "required": ["xs"],
              "properties": {"xs": {"type": "array",
                                    "items": {"type": ["integer", "null"]}}}}
    assert validate(schema, {"xs": [1, None, 3]}) == []
    assert any("missing required" in e for e in validate(schema, {}))
    assert any("expected" in e for e in validate(schema, {"xs": [1, "two"]}))
    # bool must not satisfy integer.
    assert validate({"type": "integer"}, True) != []
