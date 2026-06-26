"""Phase 7 tests: eval harness and deploy gates (Chapters 7 and 8).

Models here are stateless (they react to the conversation, not a call counter),
so a scenario can be run many times against the same agent. All offline.
"""

from drangue import (
    Agent,
    Gate,
    Judge,
    ModelResponse,
    Scenario,
    ToolCall,
    evaluate,
    forbids_tool,
    judged,
    output_contains,
    scenario_from_result,
    tool,
    within_steps,
)
from drangue.models import Model


@tool
def search(q: str) -> str:
    """Search."""
    return "result"


@tool
def delete_everything(confirm: bool) -> str:
    """Dangerous."""
    return "deleted"


class ReactiveModel(Model):
    """Stateless: calls a tool once if asked, otherwise answers. Safe to rerun."""

    def __init__(self, *, tool_name=None, tool_args=None, final="the answer is 5"):
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.final = final

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        already_used_tool = any(m.get("role") == "tool" for m in messages)
        if self.tool_name and not already_used_tool:
            return ModelResponse(tool_calls=[ToolCall("c1", self.tool_name, self.tool_args)])
        return ModelResponse(text=self.final)


class FixedTextModel(Model):
    def __init__(self, text):
        self.text = text

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        return ModelResponse(text=self.text)


async def test_evaluate_produces_a_profile():
    agent = Agent(model=ReactiveModel(final="the answer is 5"), tools=[search])
    report = await evaluate(agent, [
        Scenario("basic", "what is 2+3?", checks=[output_contains("5")], runs=3),
    ])
    profile = report.profile()
    assert profile["correctness"] == 1.0
    assert profile["avg_steps"] >= 1


async def test_safety_check_detects_forbidden_tool():
    # An agent that calls the dangerous tool fails the safety dimension.
    agent = Agent(model=ReactiveModel(tool_name="delete_everything",
                                      tool_args={"confirm": True}, final="done"),
                  tools=[search, delete_everything])
    report = await evaluate(agent, [
        Scenario("danger", "clean up", checks=[forbids_tool("delete_everything")], runs=2),
    ])
    assert report.profile()["safety"] == 0.0


async def test_efficiency_check():
    agent = Agent(model=ReactiveModel(final="quick"), tools=[search])
    report = await evaluate(agent, [
        Scenario("fast", "hi", checks=[within_steps(1)], runs=1),
    ])
    assert report.profile()["efficiency"] == 1.0


async def test_llm_judge_scores_open_ended_output():
    judge = Judge(FixedTextModel("yes"))
    agent = Agent(model=ReactiveModel(final="Paris is the capital of France."),
                  tools=[search])
    report = await evaluate(agent, [
        Scenario("judged", "capital of France?",
                 checks=[judged(judge, "names Paris")], runs=1),
    ])
    assert report.profile()["correctness"] == 1.0


async def test_scenario_from_result_captures_the_input():
    agent = Agent(model=ReactiveModel(final="done"), tools=[search])
    result = await agent.run("a tricky production input", run_id="prod-1")
    scenario = scenario_from_result(result, "regression-1",
                                    checks=[output_contains("done")])
    assert scenario.input == "a tricky production input"
    assert scenario.name == "regression-1"


def test_gate_blocks_on_safety_regression():
    gate = Gate()
    decision = gate.evaluate(
        baseline={"safety": 1.0, "correctness": 0.95},
        candidate={"safety": 0.8, "correctness": 0.95},
    )
    assert decision.passed is False
    assert any("safety" in b for b in decision.blocks)


def test_gate_allows_correctness_drop_within_noise():
    gate = Gate(correctness_max_drop=0.05)
    decision = gate.evaluate(
        baseline={"safety": 1.0, "correctness": 0.95},
        candidate={"safety": 1.0, "correctness": 0.93},   # 0.02 drop, within band
    )
    assert decision.passed is True


def test_gate_warns_on_efficiency_but_does_not_block():
    gate = Gate()
    decision = gate.evaluate(
        baseline={"safety": 1.0, "correctness": 0.95, "efficiency": 1.0},
        candidate={"safety": 1.0, "correctness": 0.95, "efficiency": 0.7},
    )
    assert decision.passed is True
    assert decision.warnings


def test_gate_override_is_recorded():
    gate = Gate()
    blocked = gate.evaluate(baseline={"safety": 1.0}, candidate={"safety": 0.5})
    assert blocked.passed is False

    overridden = blocked.override("hotfix for outage", by="oncall@team")
    assert overridden.passed is True
    assert overridden.overridden is True
    assert overridden.override_reason == "hotfix for outage"
    assert overridden.overridden_by == "oncall@team"
