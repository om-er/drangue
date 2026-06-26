"""Phase 6 tests: per-action autonomy and durable approval (Chapter 12).

The pause/resume tests use one agent and store across the pause, the same way a
real paused run is resumed later. All offline.
"""

from drangue import Agent, Autonomy, ModelResponse, ToolCall, tool
from drangue.models import Model


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = steps
        self.i = 0
        self.calls = 0

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        self.calls += 1
        resp = self.steps[self.i]
        self.i += 1
        return resp


def _make(tool_obj, ran):
    """A model that calls the tool once, then answers."""
    return ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", tool_obj.name, {"x": 1})],
                      reasoning="this action is needed"),
        ModelResponse(text="done"),
    ])


async def test_autonomous_executes_normally():
    ran = []

    @tool
    def act(x: int) -> str:
        """Act."""
        ran.append(x)
        return "acted"

    agent = Agent(model=_make(act, ran), tools=[act],
                  autonomy=Autonomy(default="autonomous"))
    result = await agent.run("go")
    assert result.output == "done"
    assert ran == [1]


async def test_shadow_proposes_without_executing():
    ran = []

    @tool
    def act(x: int) -> str:
        """Act."""
        ran.append(x)
        return "acted"

    agent = Agent(model=_make(act, ran), tools=[act],
                  autonomy=Autonomy(modes={"act": "shadow"}))
    result = await agent.run("go")

    assert result.output == "done"
    assert ran == []   # the side effect did NOT happen
    tr = [e for e in result.events if e.type == "tool_result"][0]
    assert "shadow" in tr.payload["content"]


async def test_assisted_pauses_then_resumes_on_approval():
    ran = []

    @tool
    def act(x: int) -> str:
        """Act."""
        ran.append(x)
        return "acted"

    model = _make(act, ran)
    agent = Agent(model=model, tools=[act],
                  autonomy=Autonomy(modes={"act": "assisted"}))

    paused = await agent.run("go", run_id="r1")
    assert paused.status == "paused"
    assert ran == []                 # not executed yet
    assert model.calls == 1          # paused right after the proposal

    pending = await agent.pending_approvals("r1")
    assert len(pending) == 1
    assert pending[0]["tool"] == "act"
    assert pending[0]["reasoning"] == "this action is needed"   # the surface shows the case

    await agent.approve("r1")
    result = await agent.resume("r1")

    assert result.status == "completed"
    assert result.output == "done"
    assert ran == [1]                # executed exactly once after approval
    assert model.calls == 2          # resumed and finished


async def test_assisted_rejection_blocks_the_action():
    ran = []

    @tool
    def act(x: int) -> str:
        """Act."""
        ran.append(x)
        return "acted"

    model = _make(act, ran)
    agent = Agent(model=model, tools=[act],
                  autonomy=Autonomy(modes={"act": "assisted"}))

    await agent.run("go", run_id="r2")
    await agent.reject("r2", reason="too risky")
    result = await agent.resume("r2")

    assert result.status == "completed"
    assert ran == []                 # never executed
    tr = [e for e in result.events if e.type == "tool_result"][0]
    assert "denied" in tr.payload["content"]


async def test_per_action_modes_coexist():
    log = []

    @tool
    def safe(x: int) -> str:
        """Safe, autonomous."""
        log.append("safe")
        return "ok"

    @tool
    def risky(x: int) -> str:
        """Risky, assisted."""
        log.append("risky")
        return "ok"

    # The model calls the autonomous tool, then the assisted one.
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "safe", {"x": 1})]),
        ModelResponse(tool_calls=[ToolCall("c2", "risky", {"x": 2})]),
        ModelResponse(text="done"),
    ])
    agent = Agent(model=model, tools=[safe, risky],
                  autonomy=Autonomy(default="autonomous", modes={"risky": "assisted"}))

    paused = await agent.run("go", run_id="r3")
    assert paused.status == "paused"
    assert log == ["safe"]           # autonomous ran, assisted is waiting

    await agent.approve("r3")
    result = await agent.resume("r3")
    assert result.output == "done"
    assert log == ["safe", "risky"]
