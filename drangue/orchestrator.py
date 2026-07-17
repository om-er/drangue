"""The orchestrator: pure, deterministic decision-making.

It decides the next step purely from the folded event log. No clock, no random,
no I/O. That discipline is what lets a resumed run make exactly the same
decisions as the original (enforced and tested in Phase 2).

The orchestrator never calls a tool or a model. It returns a Step describing
what should happen; the executor is what actually makes it happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import ToolCall


@dataclass
class ModelStep:
    """Ask the model to decide what to do next."""

    seq: int
    index: int = 0   # which model call this is (0-based), for routing


@dataclass
class ToolStep:
    """Run one tool the model asked for."""

    seq: int
    call: ToolCall


@dataclass
class Done:
    """The run is complete."""

    output: str


@dataclass
class State:
    """The folded view of the log. A cache of replaying events, never stored."""

    messages: list = field(default_factory=list)   # neutral conversation history
    pending: list = field(default_factory=list)     # tool calls awaiting execution
    finished: bool = False
    output: str = ""
    next_seq: int = 0
    model_calls: int = 0
    recalled: list = field(default_factory=list)    # recalled memory items (dicts)
    recalled_done: bool = False                      # has a recall step happened?
    input: str = ""                                  # the recorded run input


def fold(events: list) -> State:
    """Reconstruct state by replaying the log. Pure function of the events."""
    messages: list = []
    current_calls: list = []
    results: dict = {}
    finished = False
    output = ""
    next_seq = 0
    model_calls = 0
    recalled: list = []
    recalled_done = False
    input = ""

    for e in events:
        next_seq = e.seq + 1
        if e.type == "run_started":
            input = e.payload.get("input", "")
            messages.append({"role": "user", "content": e.payload["input"]})
        elif e.type == "memory_recalled":
            recalled = e.payload.get("items", [])
            recalled_done = True
        elif e.type == "model_decision":
            model_calls += 1
            text = e.payload.get("text", "")
            calls = [
                ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
                for c in e.payload.get("tool_calls", [])
            ]
            assistant: dict = {"role": "assistant", "content": text}
            if calls:
                assistant["tool_calls"] = [
                    {"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls
                ]
            messages.append(assistant)
            current_calls = calls
            results = {}
            if not calls:
                finished = True
                output = text
        elif e.type == "tool_result":
            cid = e.payload["call_id"]
            results[cid] = e.payload["content"]
            messages.append({
                "role": "tool",
                "call_id": cid,
                "name": e.payload.get("name", ""),
                "content": e.payload["content"],
            })
        elif e.type == "run_finished":
            finished = True
            output = e.payload.get("output", "")

    pending = [c for c in current_calls if c.id not in results]
    return State(
        messages=messages,
        pending=pending,
        finished=finished,
        output=output,
        next_seq=next_seq,
        model_calls=model_calls,
        recalled=recalled,
        recalled_done=recalled_done,
        input=input,
    )


class Orchestrator:
    """Decides the next step from state. Deterministic by construction."""

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps

    def next(self, state: State):
        if state.finished:
            return Done(state.output)
        if state.pending:
            return ToolStep(seq=state.next_seq, call=state.pending[0])
        if state.model_calls >= self.max_steps:
            return Done(f"(stopped: reached max_steps={self.max_steps})")
        return ModelStep(seq=state.next_seq, index=state.model_calls)
