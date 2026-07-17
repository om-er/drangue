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
    started: set = field(default_factory=set)        # call_ids with a recorded
                                                     # step_started but no result yet
    finish_recorded: bool = False                     # current turn has a
                                                      # run_finished in the log
    output_schema: dict | None = None                 # recorded structured-output
                                                      # contract, if any
    schema_feedback_count: int = 0                    # corrective retries so far
    config: dict | None = None                        # recorded run config
                                                      # (max_steps, autonomy, tools)


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
    started: set = set()
    finish_recorded = False
    output_schema = None
    schema_feedback_count = 0
    config = None

    for e in events:
        next_seq = e.seq + 1
        if e.type == "run_started":
            input = e.payload.get("input", "")
            output_schema = e.payload.get("output_schema")
            config = e.payload.get("config")
            messages.append({"role": "user", "content": e.payload["input"]})
        elif e.type == "schema_feedback":
            # A recorded correction: the model's final answer missed the
            # schema, so the conversation reopens with the errors spelled out.
            messages.append({"role": "user", "content": e.payload.get("text", "")})
            finished = False
            finish_recorded = False
            output = ""
            schema_feedback_count += 1
        elif e.type == "user_message":
            # A follow-up turn: the conversation reopens and the orchestrator
            # will ask the model again.
            messages.append({"role": "user", "content": e.payload.get("text", "")})
            finished = False
            finish_recorded = False
            output = ""
        elif e.type == "step_started":
            started.add(e.payload.get("call_id"))
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
            if e.payload.get("thinking_blocks"):
                assistant["thinking_blocks"] = e.payload["thinking_blocks"]
            messages.append(assistant)
            current_calls = calls
            results = {}
            if not calls:
                finished = True
                output = text
        elif e.type == "tool_result":
            cid = e.payload["call_id"]
            results[cid] = e.payload["content"]
            started.discard(cid)
            messages.append({
                "role": "tool",
                "call_id": cid,
                "name": e.payload.get("name", ""),
                "content": e.payload["content"],
            })
        elif e.type == "run_finished":
            finished = True
            finish_recorded = True
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
        started=started,
        finish_recorded=finish_recorded,
        output_schema=output_schema,
        schema_feedback_count=schema_feedback_count,
        config=config,
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
        # The RECORDED limit wins over the live one, so a resumed run makes
        # the same decisions the original would have (config drift between
        # run and resume must not rewrite history).
        max_steps = self.max_steps
        if state.config and state.config.get("max_steps") is not None:
            max_steps = state.config["max_steps"]
        if state.model_calls >= max_steps:
            return Done(f"(stopped: reached max_steps={max_steps})")
        return ModelStep(seq=state.next_seq, index=state.model_calls)
