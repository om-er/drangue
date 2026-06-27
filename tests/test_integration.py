"""Composed integration test: several features active in one run.

Every other test exercises one feature in isolation. This one drives memory +
guardrails + autonomy + budget through a single run, so the engine loop ordering
(input-guard -> recall -> budget -> done -> autonomy -> execute) is exercised
with everything live at once. That ordering is the seam the RunContext refactor
makes easy to widen, so it is the thing worth protecting.
"""

from drangue import (
    Agent,
    Autonomy,
    Budget,
    Guardrails,
    MemoryItem,
    ModelResponse,
    ToolCall,
    tool,
)
from drangue.memory import Memory
from drangue.models import Model


class OneMemory(Memory):
    def __init__(self, item):
        self.item = item

    async def recall(self, query, *, limit=5):
        return [self.item]

    async def remember(self, item):
        return None


async def test_memory_guardrails_autonomy_and_budget_in_one_run():
    ran = []

    @tool
    def gather(x: int) -> str:
        """Autonomous, read-only."""
        ran.append("gather")
        return "telemetry"

    @tool(reversible=False)
    def remediate(x: int) -> str:
        """Assisted; must be approved."""
        ran.append("remediate")
        return "fixed"

    class Reactive(Model):
        """Stateless: gather, then remediate, then conclude. Records system."""

        def __init__(self):
            self.systems = []

        async def generate(self, *, system, messages, tools, idempotency_key=None):
            self.systems.append(system)
            done_tools = [m["name"] for m in messages if m.get("role") == "tool"]
            if "gather" not in done_tools:
                return ModelResponse(tool_calls=[ToolCall("c1", "gather", {"x": 1})])
            if "remediate" not in done_tools:
                return ModelResponse(tool_calls=[ToolCall("c2", "remediate", {"x": 2})],
                                     reasoning="metrics point to a bad index")
            return ModelResponse(text="resolved",
                                 usage={"input_tokens": 5, "output_tokens": 1})

    model = Reactive()
    agent = Agent(
        model=model,
        tools=[gather, remediate],
        instructions="You are an SRE agent.",
        memory=OneMemory(MemoryItem(key="past", value="last week: bad index on orders")),
        guardrails=Guardrails(allow={"gather", "remediate"}),
        autonomy=Autonomy(default="autonomous", modes={"remediate": "assisted"}),
        budget=Budget(max_tokens=10_000),
    )

    paused = await agent.run("orders latency is high", run_id="inc-1")

    # Memory was recalled first and injected into the model's context.
    types = [e.type for e in paused.events]
    assert types[0] == "run_started"
    assert types[1] == "memory_recalled"            # recall runs before the first model step
    assert "last week: bad index on orders" in model.systems[0]

    # The autonomous tool ran; the assisted one paused for approval.
    assert paused.status == "paused"
    assert ran == ["gather"]
    pending = await agent.pending_approvals("inc-1")
    assert pending[0]["tool"] == "remediate"
    assert pending[0]["reasoning"] == "metrics point to a bad index"

    # Approve and resume: the assisted tool runs, the run completes.
    await agent.approve("inc-1")
    done = await agent.resume("inc-1")

    assert done.status == "completed"
    assert done.output == "resolved"
    assert ran == ["gather", "remediate"]
    # Guardrails let both permitted tools through; nothing was blocked.
    assert not any("blocked" in str(e.payload.get("content", "")) for e in done.events)
