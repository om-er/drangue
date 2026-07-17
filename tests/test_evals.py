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


class FlakyModel(Model):
    """Alternates pass/fail across calls: exactly 50% over an even run count."""

    def __init__(self):
        self.calls = 0

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        self.calls += 1
        return ModelResponse(text="the answer is 5" if self.calls % 2 else "no idea")


async def test_profile_reports_the_error_bar_on_a_flaky_rate():
    # The point of running N times: a rate alone cannot say how solid it is.
    agent = Agent(model=FlakyModel(), tools=[search])
    report = await evaluate(agent, [
        Scenario("flaky", "what is 2+3?", checks=[output_contains("5")], runs=4),
    ])
    profile = report.profile()

    assert profile["correctness"] == 0.5
    # Agresti-Coull standard error at p=0.5 over 4 runs: sqrt(.5*.5/(4+z^2)).
    expected = (0.5 * 0.5 / (4 + 1.96 ** 2)) ** 0.5
    assert abs(profile["stderr"]["correctness"] - expected) < 1e-9
    assert abs(profile["noise"] - expected) < 1e-9


async def test_unanimous_small_samples_still_carry_uncertainty():
    # 3/3 passes is NOT certainty: the true rate could plausibly be ~30%. The
    # Wald formula reports zero here; the Agresti-Coull adjustment must not.
    agent = Agent(model=ReactiveModel(final="the answer is 5"), tools=[search])
    report = await evaluate(agent, [
        Scenario("steady", "what is 2+3?", checks=[output_contains("5")], runs=3),
    ])
    profile = report.profile()

    assert profile["correctness"] == 1.0
    assert profile["noise"] > 0.05          # agreement over 3 runs is a weak claim
    assert profile["stdev_steps"] == 0.0    # the runs themselves were identical

    sr = report.scenarios[0]
    assert sr.rate_stderr("correctness") > 0.05
    assert sr.stdev_tokens == 0.0


async def test_more_runs_shrink_the_error_bar():
    from drangue.evals import _rate_stderr

    assert _rate_stderr(3, 3) > _rate_stderr(30, 30) > _rate_stderr(300, 300) > 0.0
    assert _rate_stderr(0, 3) > _rate_stderr(0, 300) > 0.0


def test_gate_warns_when_noise_swamps_the_threshold():
    # A 0.05 threshold cannot resolve a difference measured to +/-0.25.
    gate = Gate(correctness_max_drop=0.05)
    baseline = {"correctness": 0.5, "safety": 1.0, "efficiency": 1.0}
    candidate = {"correctness": 0.5, "safety": 1.0, "efficiency": 1.0, "noise": 0.25}

    decision = gate.evaluate(baseline, candidate)

    assert decision.passed                      # noise informs, it does not veto
    assert not decision.blocks
    assert any("standard error" in w for w in decision.warnings)


def test_gate_is_quiet_about_noise_when_the_band_is_tight():
    gate = Gate(correctness_max_drop=0.05)
    profile = {"correctness": 1.0, "safety": 1.0, "efficiency": 1.0, "noise": 0.0}

    decision = gate.evaluate(profile, profile)

    assert decision.passed
    assert not decision.warnings


async def test_evaluate_refuses_a_run_id_collision():
    # A second evaluate() against the same store would replay the recorded
    # runs (zero model calls) and present the old profile as a fresh one.
    agent = Agent(model=ReactiveModel(final="the answer is 5"), tools=[search])
    scenarios = [Scenario("basic", "what is 2+3?", checks=[output_contains("5")])]
    await evaluate(agent, scenarios, run_id_prefix="build-1")

    try:
        await evaluate(agent, scenarios, run_id_prefix="build-1")
        assert False, "expected ValueError for a reused run_id_prefix"
    except ValueError as exc:
        assert "already exists" in str(exc)

    # A fresh prefix measures again.
    report = await evaluate(agent, scenarios, run_id_prefix="build-2")
    assert report.profile()["correctness"] == 1.0


def test_gate_blocks_when_a_measured_dimension_disappears():
    # Deleting the only safety scenario must not read as safety == 1.0.
    gate = Gate()
    baseline = {"correctness": 1.0, "safety": 1.0}
    candidate = {"correctness": 1.0}    # safety no longer measured

    decision = gate.evaluate(baseline, candidate)
    assert not decision.passed
    assert any("no longer checked" in b for b in decision.blocks)


def test_gate_warns_when_baseline_noise_swamps_the_threshold():
    gate = Gate(correctness_max_drop=0.05)
    baseline = {"correctness": 0.5, "noise": 0.25}
    candidate = {"correctness": 0.5}

    decision = gate.evaluate(baseline, candidate)
    assert decision.passed
    assert any("standard error" in w for w in decision.warnings)


async def test_judge_accepts_a_verdict_that_is_not_the_first_word():
    judge = Judge(FixedTextModel("The output satisfies the criterion, so yes."))
    assert await judge.yes_no("does it?") is True

    judge = Judge(FixedTextModel("It does not satisfy it: no."))
    assert await judge.yes_no("does it?") is False

    # No recognizable verdict scores as no, never as yes.
    judge = Judge(FixedTextModel("cannot tell"))
    assert await judge.yes_no("does it?") is False


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


def test_gate_override_reaches_the_record_sink():
    # "Recorded" has to mean something left the process. The sink is the seam.
    written = []
    gate = Gate()
    blocked = gate.evaluate(baseline={"safety": 1.0}, candidate={"safety": 0.5})

    blocked.override("hotfix for outage", by="oncall@team", record=written.append)

    assert len(written) == 1
    entry = written[0]
    assert entry["overridden"] is True
    assert entry["override_reason"] == "hotfix for outage"
    assert entry["overridden_by"] == "oncall@team"
    # The record carries what was overridden, not just that it was.
    assert any("safety" in b for b in entry["blocks"])


def test_gate_override_demands_a_reason():
    gate = Gate()
    blocked = gate.evaluate(baseline={"safety": 1.0}, candidate={"safety": 0.5})

    for empty in ("", "   "):
        try:
            blocked.override(empty, by="oncall@team")
            assert False, "expected ValueError for an override with no reason"
        except ValueError as exc:
            assert "needs a reason" in str(exc)


def test_gate_override_record_is_json_shaped():
    import json

    gate = Gate()
    blocked = gate.evaluate(baseline={"safety": 1.0}, candidate={"safety": 0.5})
    record = blocked.override("shipping anyway", by="me").to_record()

    # A sink is usually a file or a table; the record must survive the trip.
    assert json.loads(json.dumps(record)) == record
