"""Human-in-the-loop and progressive rollout (Chapter 12).

Autonomy is granted to actions, not to the agent. Each tool runs in one of three
modes, and the agent can be autonomous on one action while gated on another:

  - shadow      propose the action, do not execute it, record the proposal. This
                builds a track record against reality without risk.
  - assisted    propose, then wait for a human to approve before executing.
  - autonomous  execute, with the run reviewable afterwards.

Assisted mode is a durable pause. When an assisted action is reached, the engine
records an `approval_requested` event (carrying the agent's reasoning, so the
human approves a case, not a bare action) and stops. A later `approval_granted`
or `approval_denied` event, appended out of band, lets the run resume by replay.
The approval is just another fact in the log, which is what makes the pause
survive a process restart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

SHADOW = "shadow"
ASSISTED = "assisted"
AUTONOMOUS = "autonomous"


@dataclass
class Autonomy:
    default: str = AUTONOMOUS
    modes: dict = field(default_factory=dict)   # tool_name -> mode

    def mode_for(self, name: str) -> str:
        return self.modes.get(name, self.default)


def shadow_result(call) -> str:
    return json.dumps({
        "shadow": True,
        "proposed": {"tool": call.name, "arguments": call.arguments},
        "note": "not executed (shadow mode)",
    })


def denied_result(call, reason: str = "") -> str:
    return json.dumps({
        "ok": False,
        "tool": call.name,
        "error": {"category": "denied", "message": reason or "action was not approved"},
    })


def approval_decision(events, call_id: str):
    """Return ('granted'|'denied'|'pending', reason)."""
    for e in events:
        if e.payload.get("call_id") != call_id:
            continue
        if e.type == "approval_granted":
            return "granted", None
        if e.type == "approval_denied":
            return "denied", e.payload.get("reason", "")
    return "pending", None


def has_approval_request(events, call_id: str) -> bool:
    return any(e.type == "approval_requested" and e.payload.get("call_id") == call_id
               for e in events)


def pending_approvals(events) -> list:
    """Approval requests that have not yet been granted or denied."""
    requested: dict = {}
    resolved: set = set()
    for e in events:
        if e.type == "approval_requested":
            requested[e.payload["call_id"]] = e.payload
        elif e.type in ("approval_granted", "approval_denied"):
            resolved.add(e.payload.get("call_id"))
    return [payload for cid, payload in requested.items() if cid not in resolved]


def first_pending_call(events):
    pend = pending_approvals(events)
    return pend[0]["call_id"] if pend else None


def next_seq(events) -> int:
    return max((e.seq for e in events), default=-1) + 1


def last_reasoning(events) -> dict:
    for e in reversed(events):
        if e.type == "model_decision":
            return {"reasoning": e.payload.get("reasoning"), "text": e.payload.get("text", "")}
    return {"reasoning": None, "text": ""}
