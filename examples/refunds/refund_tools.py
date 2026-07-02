"""Refund tools: five readers and two spenders, with the policy in the code.

The design decision that carries this example: there are TWO refund tools,
and the boundary between them is an `if` statement, not a prompt.

  - `refund_small` settles below the €100 cap autonomously — and refuses
    anything at or above it, or anything with fraud signals, in the tool
    body. A prompt-injected model asking it for €840 gets a refusal.
  - `refund` has no cap and never executes on its own: the Agent runs it in
    assisted mode, so it pauses for a human every time.

Both spenders declare an `idempotency_key`. The fake payment provider
deduplicates on it the way a real PSP's refund endpoint does: a retry with
the same key returns the original refund instead of paying twice.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path

from drangue import tool

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
ORDERS = json.loads((_FIXTURES / "orders.json").read_text())

AUTONOMOUS_CAP_EUR = 100.0


def _ledger() -> Path:
    return Path(os.environ.get(
        "PSP_LEDGER", Path(tempfile.gettempdir()) / "drangue_psp_refunds.jsonl"))


def _psp_refund(order_id: str, amount_eur: float, reason: str, key: str) -> dict:
    """The fake payment provider. Idempotent on the key, like the real thing."""
    ledger = _ledger()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            entry = json.loads(line)
            if key and entry["idempotency_key"] == key:
                return {**entry, "deduplicated": True}
    entry = {"psp_ref": f"re_{uuid.uuid4().hex[:8]}", "order_id": order_id,
             "amount_eur": amount_eur, "reason": reason, "idempotency_key": key}
    with ledger.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def _order(order_id: str) -> dict | None:
    return ORDERS.get(order_id.upper().strip())


def _refused(reason: str) -> str:
    return json.dumps({"ok": False, "refused": reason})


@tool(timeout=5)
def order_lookup(order_id: str) -> str:
    """Look up an order: total, status, and items."""
    o = _order(order_id)
    if not o:
        return _refused(f"unknown order {order_id}")
    return json.dumps({"order_id": order_id, "total_eur": o["total_eur"],
                       "status": o["status"], "items": o["items"]})


@tool(timeout=5)
def payment_status(order_id: str) -> str:
    """List the payment captures for an order (amount, time, card)."""
    o = _order(order_id)
    if not o:
        return _refused(f"unknown order {order_id}")
    return json.dumps({"order_id": order_id, "captures": o["captures"]})


@tool(timeout=5)
def delivery_events(order_id: str) -> str:
    """The carrier's delivery events for an order, and whether it was signed for."""
    o = _order(order_id)
    if not o:
        return _refused(f"unknown order {order_id}")
    return json.dumps({"order_id": order_id, **o["delivery"]})


@tool(timeout=5)
def customer_history(order_id: str) -> str:
    """The ordering customer's history: account age, prior orders, chargebacks."""
    o = _order(order_id)
    if not o:
        return _refused(f"unknown order {order_id}")
    return json.dumps({"order_id": order_id, **o["customer"]})


@tool(timeout=5)
def fraud_signals(order_id: str) -> str:
    """Fraud markers on an order. Any flag routes the refund to the gated path."""
    o = _order(order_id)
    if not o:
        return _refused(f"unknown order {order_id}")
    return json.dumps({"order_id": order_id, "flags": o["fraud_flags"],
                       "count": len(o["fraud_flags"])})


@tool(timeout=5)
def refund_small(order_id: str, amount_eur: float, reason: str,
                 idempotency_key: str = "") -> str:
    """Settle a refund below the €100 autonomous cap, if the evidence is clean.
    The cap and the fraud check are enforced here, in code."""
    o = _order(order_id)
    if not o:
        return _refused(f"unknown order {order_id}")
    if amount_eur >= AUTONOMOUS_CAP_EUR:
        return _refused(f"€{amount_eur:.2f} is at or above the €{AUTONOMOUS_CAP_EUR:.0f} "
                        "autonomous cap; use `refund`, which requires approval")
    if o["fraud_flags"]:
        return _refused("fraud signals present; route via the approval-gated `refund`")
    if amount_eur > o["total_eur"]:
        return _refused(f"€{amount_eur:.2f} exceeds the order total €{o['total_eur']:.2f}")
    entry = _psp_refund(order_id, amount_eur, reason, idempotency_key)
    return json.dumps({"ok": True, "executed": True, **entry})


@tool(timeout=5)
def refund(order_id: str, amount_eur: float, reason: str,
           idempotency_key: str = "") -> str:
    """Settle a refund of any amount. Runs in assisted mode: a human approves
    the case before this executes."""
    o = _order(order_id)
    if not o:
        return _refused(f"unknown order {order_id}")
    if amount_eur > o["total_eur"]:
        return _refused(f"€{amount_eur:.2f} exceeds the order total €{o['total_eur']:.2f}")
    entry = _psp_refund(order_id, amount_eur, reason, idempotency_key)
    return json.dumps({"ok": True, "executed": True, **entry})


# Support tickets are attacker-controlled text aimed at an agent that can move
# money. The scan runs before the model reasons over the ticket; everything
# downstream (the cap, the gate) assumes it missed something.
_INJECTION = re.compile(
    r"ignore (all )?(previous|prior|your)( |-)?(instructions|rules|policy)?"
    r"|disregard .*(instructions|policy)"
    r"|you are now"
    r"|system prompt",
    re.IGNORECASE)


def ticket_injection_scan(text: str) -> str | None:
    if _INJECTION.search(text):
        return "ticket contains prompt-injection patterns; routed to a human"
    return None


ALL_TOOLS = [order_lookup, payment_status, delivery_events, customer_history,
             fraud_signals, refund_small, refund]
