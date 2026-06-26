"""Phase 1 tests: timing, cost, the trace tree, reasoning, and tracers.

All offline with a scripted model.
"""

import contextlib
import io

from drangue import Agent, ModelResponse, ToolCall, tool
from drangue.models import Model


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = steps
        self.i = 0

    async def generate(self, *, system, messages, tools):
        resp = self.steps[self.i]
        self.i += 1
        return resp


def _model():
    return ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("t1", "add", {"a": 2, "b": 3})],
                      usage={"input_tokens": 10, "output_tokens": 5}),
        ModelResponse(text="5", reasoning="2 plus 3 is 5",
                      usage={"input_tokens": 20, "output_tokens": 3}),
    ])


async def test_steps_carry_timing():
    result = await Agent(model=_model(), tools=[add]).run("2 + 3")
    steps = [e for e in result.events if e.type in ("model_decision", "tool_result")]
    assert steps  # there were steps
    assert all(e.ts is not None for e in steps)
    assert all(e.duration_ms is not None and e.duration_ms >= 0 for e in steps)


async def test_result_trace_is_a_navigable_tree():
    result = await Agent(model=_model(), tools=[add]).run("2 + 3")
    tree = result.trace

    assert tree.name == "run"
    assert tree.attrs["output"] == "5"
    assert [c.name for c in tree.children] == ["model", "tool", "model"]
    assert all(c.duration_ms is not None for c in tree.children)
    # The tree renders without error (used for human inspection).
    assert "run" in tree.render()


async def test_reasoning_is_captured():
    result = await Agent(model=_model(), tools=[add]).run("2 + 3")
    decisions = [e for e in result.events if e.type == "model_decision"]
    assert decisions[-1].payload["reasoning"] == "2 plus 3 is 5"


async def test_usage_and_latency_totals():
    result = await Agent(model=_model(), tools=[add]).run("2 + 3")
    assert result.usage == {"input_tokens": 30, "output_tokens": 8}
    assert result.latency_ms >= 0


async def test_console_tracer_prints_when_trace_true():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        await Agent(model=_model(), tools=[add]).run("2 + 3", trace=True)
    out = buf.getvalue()
    assert "add(" in out
    assert "5" in out


async def test_silent_by_default():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        await Agent(model=_model(), tools=[add]).run("2 + 3")
    assert buf.getvalue() == ""
