"""Recorded run config: a resumed run decides as the original was configured.

Config drift between run and resume (max_steps lowered, autonomy flipped,
tools swapped) must not rewrite a run's history. All offline.
"""

import warnings

from drangue import Agent, Autonomy, InMemoryStore, ModelResponse, ToolCall, tool
from drangue.models import Model


@tool
def act(x: int) -> str:
    """Act."""
    return "acted"


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = steps
        self.i = 0
        self.calls = 0

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        self.calls += 1
        step = self.steps[self.i]
        self.i += 1
        if step == "CRASH":
            raise RuntimeError("process died")
        return step


async def test_recorded_max_steps_wins_over_a_lowered_live_value():
    store = InMemoryStore()
    crashing = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "act", {"x": 1})]),
        "CRASH",
    ])
    agent_a = Agent(model=crashing, tools=[act], store=store, max_steps=5)
    try:
        await agent_a.run("go", run_id="r")
    except RuntimeError:
        pass

    # The resuming agent is configured tighter; the recorded limit of 5 must
    # still govern, so the run completes instead of stopping at max_steps=1.
    recovering = ScriptedModel([ModelResponse(text="done")])
    agent_b = Agent(model=recovering, tools=[act], store=store, max_steps=1)
    result = await agent_b.run("go", run_id="r")

    assert result.output == "done"
    assert "max_steps" not in result.output


async def test_recorded_assisted_mode_survives_an_autonomy_free_resume():
    store = InMemoryStore()
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "act", {"x": 1})]),
        ModelResponse(text="done"),
    ])
    agent_a = Agent(model=model, tools=[act], store=store,
                    autonomy=Autonomy(modes={"act": "assisted"}))
    paused = await agent_a.run("go", run_id="p")
    assert paused.status == "paused"

    # An agent with NO live autonomy resumes: the recorded assisted gate must
    # hold, so the still-unapproved action stays paused rather than executing.
    agent_b = Agent(model=model, tools=[act], store=store)
    still = await agent_b.resume("p")
    assert still.status == "paused"

    await agent_b.approve("p")
    result = await agent_b.resume("p")
    assert result.output == "done"


async def test_recorded_no_autonomy_ignores_a_later_assisted_flip():
    store = InMemoryStore()
    crashing = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "act", {"x": 1})]),
        "CRASH",
    ])
    agent_a = Agent(model=crashing, tools=[act], store=store)   # no autonomy
    try:
        await agent_a.run("go", run_id="n")
    except RuntimeError:
        pass

    # Flipping the tool to assisted AFTER the fact must not retroactively
    # gate a run that was recorded without autonomy.
    recovering = ScriptedModel([ModelResponse(text="done")])
    agent_b = Agent(model=recovering, tools=[act], store=store,
                    autonomy=Autonomy(modes={"act": "assisted"}))
    result = await agent_b.run("go", run_id="n")
    assert result.status == "completed"
    assert result.output == "done"


async def test_tool_set_drift_warns_but_proceeds_with_live_tools():
    store = InMemoryStore()
    model = ScriptedModel([ModelResponse(text="hi")])
    await Agent(model=model, tools=[act], store=store).run("go", run_id="t")

    @tool
    def other(x: int) -> str:
        """A different tool."""
        return "other"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        again = await Agent(model=ScriptedModel([ModelResponse(text="hi")]),
                            tools=[other], store=store).run("go", run_id="t")
    assert again.output == "hi"
    assert any("recorded with tools" in str(w.message) for w in caught)
