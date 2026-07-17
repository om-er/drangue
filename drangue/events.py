"""The event log: the source of truth for a run.

Every run is a sequence of Events. The agent's state is the result of folding
that sequence (see orchestrator.fold). Keeping the log as plain, JSON-shaped
records is deliberate: in Phase 2 the same records persist to a durable store
and a resumed run replays them to reconstruct state exactly.

Each executed step also carries timing (`ts`, `duration_ms`), recorded by the
executor as a fact. Because timing lives in the log, a run is fully
reconstructable from the outside: see `Result.trace`.

Event types:
    run_started     payload: {input}
    user_message    payload: {text}   a follow-up turn into a completed run;
                    reopens the conversation and the model is asked again
    schema_feedback payload: {text}   recorded correction when a structured
                    run's final answer missed its schema; the model is re-asked
    model_decision  payload: {text, tool_calls, usage, reasoning, model, stop_reason}
    tool_result     payload: {call_id, name, content}
    run_finished    payload: {output}
    run_failed      payload: {error, category}   step crashed; resume retries it
    step_started    payload: {call_id, tool}     intent marker appended before an
                    irreversible tool executes, so a crash between the side
                    effect and its record is detected instead of re-executed

One EPHEMERAL type flows through `agent.stream` but is never appended to the
store and never appears on replay:
    model_delta     payload: {text}   a live token chunk from a streaming model
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .budget import cost_from_events as _cost_from_events
from .rollout import pending_approvals as _pending_approvals


@dataclass
class Event:
    seq: int                          # position in the log; basis for idempotency
    type: str
    payload: dict = field(default_factory=dict)
    ts: float | None = None           # wall-clock start, recorded by the executor
    duration_ms: float | None = None  # how long the step took


@dataclass
class Span:
    """A node in a run's trace tree. Built from the log, no tracer required."""

    name: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    ts: float | None = None
    duration_ms: float | None = None

    def render(self, indent: int = 0) -> str:
        pad = "  " * indent
        dur = f"  {self.duration_ms:.1f}ms" if self.duration_ms is not None else ""
        lines = [f"{pad}{self.name}{dur}"]
        for child in self.children:
            lines.append(child.render(indent + 1))
        return "\n".join(lines)


@dataclass
class Result:
    """The outcome of a run, plus the log it came from."""

    output: str
    messages: list = field(default_factory=list)   # neutral conversation history
    events: list = field(default_factory=list)      # the full event log

    @property
    def steps(self) -> int:
        """Number of model calls in the run."""
        return sum(1 for e in self.events if e.type == "model_decision")

    @property
    def usage(self) -> dict:
        """Total token usage across the run.

        Always carries `input_tokens` and `output_tokens`; when a step
        reported cached-prompt counts (`cache_creation_input_tokens`,
        `cache_read_input_tokens` — disjoint from `input_tokens`), those
        totals ride alongside.
        """
        totals: dict = {"input_tokens": 0, "output_tokens": 0}
        for e in self.events:
            u = e.payload.get("usage") if e.type == "model_decision" else None
            if u:
                for k, v in u.items():
                    if isinstance(v, (int, float)):
                        totals[k] = totals.get(k, 0) + v
        return totals

    def cost(self, prices: dict, *, strict: bool = True) -> float:
        """Total dollar cost of the run, given a price table.

        A method rather than a property because cost is not knowable from the
        log alone: prices vary by provider and change over time, so drangue
        ships no table. Pass the same `prices` you give a Budget and the two
        agree by construction — both compute from the recorded per-step model
        and usage.

        Raises if a step used a model the table does not cover; see
        `budget.cost_from_events` for the reasoning and the strict=False escape.
        """
        return _cost_from_events(self.events, prices, strict=strict)

    @property
    def latency_ms(self) -> float:
        """Total time spent in model and tool steps."""
        return sum(e.duration_ms or 0.0 for e in self.events)

    @property
    def status(self) -> str:
        """completed, paused (awaiting approval), failed, or running."""
        if any(e.type == "run_finished" for e in self.events):
            return "completed"
        if self.pending_approvals:
            return "paused"
        if self.events and self.events[-1].type == "run_failed":
            return "failed"
        return "running"

    @property
    def error(self) -> str | None:
        """The recorded failure when status is 'failed', else None."""
        if self.events and self.events[-1].type == "run_failed":
            return self.events[-1].payload.get("error")
        return None

    @property
    def output_parsed(self):
        """The validated object from a structured run, else None.

        Set when the run had an `output_schema` and the final answer
        satisfied it (`run_finished` carries `structured: true`). A
        structured run that exhausted its corrections finishes with plain
        text and None here, so the miss is visible rather than guessed at.
        """
        for e in reversed(self.events):
            if e.type == "run_finished":
                if e.payload.get("structured"):
                    return json.loads(e.payload.get("output", "null"))
                return None
        return None

    @property
    def truncated(self) -> bool:
        """True when the final answer was cut off by the max_tokens ceiling.

        A truncated answer still completes the run; this flag is how a caller
        distinguishes "the model finished" from "the model ran out of room".
        """
        for e in reversed(self.events):
            if e.type == "model_decision":
                return e.payload.get("stop_reason") in ("max_tokens", "length")
        return False

    @property
    def pending_approvals(self) -> list:
        """Approval requests awaiting a human decision, with the agent's case."""
        return _pending_approvals(self.events)

    @property
    def trace(self) -> Span:
        """A navigable span tree reconstructed from the persisted log.

        This is the canonical, source-of-truth trace: it needs no tracer to be
        configured and is always faithful to what actually ran.
        """
        root = Span(name="run", attrs={"output": self.output},
                    duration_ms=self.latency_ms)
        for e in self.events:
            if e.type == "model_decision":
                root.children.append(Span(name="model", attrs=dict(e.payload),
                                          ts=e.ts, duration_ms=e.duration_ms))
            elif e.type == "tool_result":
                root.children.append(Span(name="tool", attrs=dict(e.payload),
                                          ts=e.ts, duration_ms=e.duration_ms))
        return root
