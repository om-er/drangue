"""Tests for the event-sourced engine: log shape and replay foundation.

These verify the Phase 0 architecture (orchestrator + executor + log) and the
in-memory replay that Phase 2 turns into crash recovery.
"""

from drangue import Agent, ModelResponse, ToolCall, tool
from drangue.models import Model


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class CountingModel(Model):
    def __init__(self, steps):
        self.steps = steps
        self.i = 0
        self.calls = 0

    async def generate(self, *, system, messages, tools):
        self.calls += 1
        resp = self.steps[self.i]
        self.i += 1
        return resp


def _two_step_model():
    return CountingModel([
        ModelResponse(tool_calls=[ToolCall("t1", "add", {"a": 2, "b": 3})]),
        ModelResponse(text="5"),
    ])


async def test_event_log_has_expected_shape():
    agent = Agent(model=_two_step_model(), tools=[add])
    result = await agent.run("2 + 3", run_id="r1")

    shape = [(e.seq, e.type) for e in result.events]
    assert shape == [
        (0, "run_started"),
        (1, "model_decision"),
        (2, "tool_result"),
        (3, "model_decision"),
        (4, "run_finished"),
    ]


async def test_completed_run_replays_without_recalling_model():
    model = _two_step_model()
    agent = Agent(model=model, tools=[add])

    first = await agent.run("2 + 3", run_id="same-id")
    assert first.output == "5"
    assert model.calls == 2

    # Same run_id against the same store: the log is complete, so the engine
    # replays it and returns the result without calling the model again.
    second = await agent.run("2 + 3", run_id="same-id")
    assert second.output == "5"
    assert model.calls == 2


async def test_usage_totals_accumulate_from_the_log():
    model = CountingModel([
        ModelResponse(tool_calls=[ToolCall("t1", "add", {"a": 1, "b": 1})],
                      usage={"input_tokens": 10, "output_tokens": 5}),
        ModelResponse(text="2", usage={"input_tokens": 20, "output_tokens": 3}),
    ])
    agent = Agent(model=model, tools=[add])
    result = await agent.run("1 + 1")
    assert result.usage == {"input_tokens": 30, "output_tokens": 8}
