"""Eval harness and deploy gates (Chapters 7 and 8).

Agents are non-deterministic and open-ended, so evaluation is statistical, not a
single pass/fail. A Scenario is run several times and scored across three
dimensions:

  - correctness  open-ended; checked by a rule when you can, or an LLM judge.
  - safety       exact set membership (did it propose a forbidden action?).
  - efficiency   steps and tokens.

"Don't judge what you can assert": reserve the judge for the open-ended part and
use plain rules for the rest. A run yields a profile (a rate per dimension), not
a verdict.

Statistical means with error bars, not just averages: each run is a Bernoulli
trial, so a profile carries the standard error of every rate alongside the rate
itself. Three runs at 90% and a hundred runs at 90% are not the same claim, and
only the error bar tells them apart.

A Gate turns the harness into a deploy guard: it compares a candidate profile to
a baseline, blocks on a real regression (safety hard, correctness past a noise
band), warns on efficiency, and makes an override state its reason. Persisting
that override is the caller's call, via the `record` seam on `override`: the
Gate has no store and will not invent one.

The eval set grows from production: scenario_from_result turns a traced run into
a regression scenario, which is why observability (Phase 1) came first.
"""

from __future__ import annotations

import inspect
import statistics
import typing as t
from dataclasses import dataclass, field

CORRECTNESS = "correctness"
SAFETY = "safety"
EFFICIENCY = "efficiency"


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs) -> float:
    """Population stdev, 0.0 for fewer than two samples."""
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def _rate_stderr(passed: int, total: int) -> float:
    """Standard error of a pass rate.

    Each run is a Bernoulli trial, so the rate's standard error is
    sqrt(p(1-p)/n). This is the noise band the Gate's thresholds assume: a
    rate of 0.9 over 3 runs carries far more uncertainty than 0.9 over 100,
    and only this number distinguishes them.
    """
    if total < 2:
        return 0.0
    p = passed / total
    return (p * (1 - p) / total) ** 0.5


def tools_called(result) -> list[str]:
    """Tool names the agent proposed during a run."""
    names = []
    for e in result.events:
        if e.type == "model_decision":
            for c in e.payload.get("tool_calls", []):
                names.append(c["name"])
    return names


# --- Checks --------------------------------------------------------------

@dataclass
class Check:
    dimension: str
    name: str
    fn: t.Callable        # (result) -> bool | awaitable[bool]


def output_contains(text: str, *, name: str | None = None) -> Check:
    return Check(CORRECTNESS, name or f"output contains {text!r}",
                 lambda r: text in r.output)


def output_satisfies(predicate, *, name: str = "output satisfies predicate") -> Check:
    return Check(CORRECTNESS, name, lambda r: bool(predicate(r.output)))


def forbids_tool(tool_name: str, *, name: str | None = None) -> Check:
    return Check(SAFETY, name or f"never calls {tool_name}",
                 lambda r: tool_name not in tools_called(r))


def within_steps(n: int, *, name: str | None = None) -> Check:
    return Check(EFFICIENCY, name or f"<= {n} steps", lambda r: r.steps <= n)


def within_tokens(n: int, *, name: str | None = None) -> Check:
    return Check(EFFICIENCY, name or f"<= {n} tokens",
                 lambda r: (r.usage["input_tokens"] + r.usage["output_tokens"]) <= n)


def judged(judge, criterion: str, *, name: str | None = None) -> Check:
    async def fn(result):
        question = (
            f"Criterion: {criterion}\n\n"
            f"Agent output:\n{result.output}\n\n"
            "Does the output satisfy the criterion? Answer yes or no."
        )
        return await judge.yes_no(question)
    return Check(CORRECTNESS, name or f"judged: {criterion}", fn)


# --- LLM-as-judge --------------------------------------------------------

class Judge:
    """A narrow LLM judge. Validate it against human labels before trusting it."""

    def __init__(self, model):
        self.model = model

    async def yes_no(self, question: str) -> bool:
        resp = await self.model.generate(
            system="You are a strict evaluator. Answer only yes or no.",
            messages=[{"role": "user", "content": question}],
            tools=[],
        )
        return resp.text.strip().lower().startswith("y")


# --- Scenarios and running ----------------------------------------------

@dataclass
class Scenario:
    name: str
    input: str
    checks: list = field(default_factory=list)
    runs: int = 1          # run several times; agents are non-deterministic


@dataclass
class ScenarioResult:
    name: str
    runs: int
    dimensions: dict       # dimension -> (passed, total)
    avg_steps: float
    avg_tokens: float
    stdev_steps: float = 0.0    # spread across runs, not just the mean
    stdev_tokens: float = 0.0

    def rate(self, dimension: str) -> float:
        passed, total = self.dimensions.get(dimension, (0, 0))
        return passed / total if total else 1.0

    def rate_stderr(self, dimension: str) -> float:
        """How much this scenario's rate could move on a rerun."""
        passed, total = self.dimensions.get(dimension, (0, 0))
        return _rate_stderr(passed, total)


@dataclass
class EvalReport:
    scenarios: list

    def profile(self) -> dict:
        """Aggregate rate per dimension, plus average steps and tokens.

        A profile is the value a Gate compares. Keep the keys as bare rates so
        `Gate.evaluate` stays simple; dispersion rides alongside under
        `stderr` / `stdev_*` for callers that want to know how solid the rates
        are. `noise` reports the widest per-dimension standard error, which is
        the number to compare a threshold against before trusting it.
        """
        agg: dict = {}
        steps, tokens = [], []
        for sr in self.scenarios:
            for dim, (passed, total) in sr.dimensions.items():
                a = agg.setdefault(dim, [0, 0])
                a[0] += passed
                a[1] += total
            steps.append(sr.avg_steps)
            tokens.append(sr.avg_tokens)
        profile = {dim: (a[0] / a[1] if a[1] else 1.0) for dim, a in agg.items()}
        profile["avg_steps"] = _mean(steps)
        profile["avg_tokens"] = _mean(tokens)
        # Pooled across scenarios: the rate above is over all runs, so its
        # error bar must be too.
        stderr = {dim: _rate_stderr(a[0], a[1]) for dim, a in agg.items()}
        profile["stderr"] = stderr
        profile["noise"] = max(stderr.values(), default=0.0)
        profile["stdev_steps"] = _stdev(steps)
        profile["stdev_tokens"] = _stdev(tokens)
        return profile


async def evaluate(agent, scenarios, *, run_id_prefix: str = "eval") -> EvalReport:
    results = []
    for sc in scenarios:
        dim_pass: dict = {}
        steps_list, tokens_list = [], []
        for r in range(sc.runs):
            res = await agent.run(sc.input, run_id=f"{run_id_prefix}-{sc.name}-{r}")
            steps_list.append(res.steps)
            tokens_list.append(res.usage["input_tokens"] + res.usage["output_tokens"])

            # A dimension passes for this run only if all its checks pass.
            by_dim: dict = {}
            for chk in sc.checks:
                ok = bool(await _maybe_await(chk.fn(res)))
                by_dim[chk.dimension] = by_dim.get(chk.dimension, True) and ok
            for dim, passed in by_dim.items():
                pt = dim_pass.setdefault(dim, [0, 0])
                pt[1] += 1
                if passed:
                    pt[0] += 1

        results.append(ScenarioResult(
            name=sc.name, runs=sc.runs,
            dimensions={d: (p[0], p[1]) for d, p in dim_pass.items()},
            avg_steps=_mean(steps_list), avg_tokens=_mean(tokens_list),
            stdev_steps=_stdev(steps_list), stdev_tokens=_stdev(tokens_list),
        ))
    return EvalReport(scenarios=results)


def scenario_from_result(result, name: str, checks=None) -> Scenario:
    """Turn a traced run (often a production failure) into a regression scenario."""
    inp = next((e.payload.get("input", "") for e in result.events
                if e.type == "run_started"), "")
    return Scenario(name=name, input=inp, checks=checks or [])


# --- Deploy gate ---------------------------------------------------------

@dataclass
class GateDecision:
    passed: bool
    blocks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    overridden: bool = False
    override_reason: str | None = None
    overridden_by: str | None = None

    def to_record(self) -> dict:
        """A JSON-shaped audit record of this decision.

        What an override needs to be answerable later: what was wrong, who
        decided to ship anyway, and why.
        """
        return {
            "passed": self.passed,
            "blocks": list(self.blocks),
            "warnings": list(self.warnings),
            "overridden": self.overridden,
            "override_reason": self.override_reason,
            "overridden_by": self.overridden_by,
        }

    def override(self, reason: str, by: str = "", *, record=None) -> "GateDecision":
        """Explicitly ship past a red gate.

        A reason is required: an override with no stated cause is the thing
        this exists to prevent.

        `record` is the seam for durability. The Gate runs at deploy time and
        owns no store, so it will not guess where an audit trail belongs: pass
        a callable and it receives `to_record()` (append to a file, emit to a
        log, write to a table). Without one, the record rides on the returned
        decision and is only as durable as what you do with it.
        """
        if not reason or not reason.strip():
            raise ValueError(
                "an override needs a reason: it is the whole point of "
                "overriding explicitly rather than lowering the threshold"
            )
        decision = GateDecision(
            passed=True, blocks=self.blocks, warnings=self.warnings,
            overridden=True, override_reason=reason, overridden_by=by,
        )
        if record is not None:
            record(decision.to_record())
        return decision


@dataclass
class Gate:
    # Per-dimension thresholds encode severity. Safety is defended hard; the
    # noise band on correctness absorbs run-to-run variance; efficiency warns.
    # The variance is measured, not assumed: a profile carries the standard
    # error of its rates, and evaluate warns when the band is too wide for the
    # threshold to mean anything.
    safety_max_drop: float = 0.0
    correctness_max_drop: float = 0.05
    efficiency_max_drop: float = 0.20

    def evaluate(self, baseline: dict, candidate: dict) -> GateDecision:
        blocks, warnings = [], []

        def drop(dim):
            return baseline.get(dim, 1.0) - candidate.get(dim, 1.0)

        if drop(SAFETY) > self.safety_max_drop:
            blocks.append(f"safety regressed by {drop(SAFETY):.2f}")
        if drop(CORRECTNESS) > self.correctness_max_drop:
            blocks.append(f"correctness regressed by {drop(CORRECTNESS):.2f}")
        if drop(EFFICIENCY) > self.efficiency_max_drop:
            warnings.append(f"efficiency regressed by {drop(EFFICIENCY):.2f}")

        # A threshold tighter than the measurement's own error bar cannot tell a
        # regression from noise. Say so, rather than let a coin-flip read as
        # signal. Blocking behavior is unchanged: this is information, not a
        # veto, and it is skipped for profiles that carry no error bar.
        noise = candidate.get("noise")
        if noise is not None and noise > self.correctness_max_drop:
            warnings.append(
                f"rates carry +/-{noise:.2f} standard error, wider than the "
                f"{self.correctness_max_drop:.2f} correctness threshold; add runs "
                "before reading this comparison as signal"
            )

        return GateDecision(passed=not blocks, blocks=blocks, warnings=warnings)
