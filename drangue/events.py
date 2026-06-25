"""The event log: the source of truth for a run.

Every run is a sequence of Events. The agent's state is the result of folding
that sequence (see orchestrator.fold). Keeping the log as plain, JSON-shaped
records is deliberate: in Phase 2 the same records persist to a durable store
and a resumed run replays them to reconstruct state exactly.

Event types in Phase 0:
    run_started     payload: {input}
    model_decision  payload: {text, tool_calls: [{id, name, arguments}], usage}
    tool_result     payload: {call_id, name, content}
    run_finished    payload: {output}
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Event:
    seq: int                      # position in the run's log; basis for idempotency
    type: str
    payload: dict = field(default_factory=dict)
    ts: str | None = None         # recorded as a fact by the executor (Phase 1)


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
        """Total token usage across the run (populated once adapters report it)."""
        inp = out = 0
        for e in self.events:
            u = e.payload.get("usage") if e.type == "model_decision" else None
            if u:
                inp += u.get("input_tokens", 0)
                out += u.get("output_tokens", 0)
        return {"input_tokens": inp, "output_tokens": out}
