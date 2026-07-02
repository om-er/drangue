"""A refund agent with a hard ceiling: settle small, escalate large.

The policy manual's rule: refunds under €100 with clean evidence settle
immediately, anything above or fraud-flagged needs a senior. Here it is
expressed so that it is provable rather than hoped for: the cap lives in
the tool body, the big-refund tool always pauses for approval, and every
disposition is an event in a durable log.

Live use needs an ANTHROPIC_API_KEY; the orders themselves come from
fixtures/, so there is no other infrastructure. For a fully scripted run
with no API key, see offline_demo.py.

    python examples/refunds/refund_agent.py \
        "I was charged twice for order ORD-88412"
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime

from drangue import Agent, Autonomy, Guardrails, SQLiteStore

from refund_tools import ALL_TOOLS, ticket_injection_scan

INSTRUCTIONS = """You are a refund agent for an e-commerce store. For each
ticket: verify the order, the payment captures, the delivery events, and the
fraud signals before deciding anything.

Policy: refunds under €100 with clean evidence settle via refund_small.
Anything at or above €100, or with any fraud flag, goes via refund, which a
senior must approve: build the complete case in your reasoning (evidence,
fraud markers, policy basis, and your recommendation, which may be to deny),
because that case is what the approver reads. Never refund more than was
captured. Be concise and specific."""


def build_agent(model="claude-opus-4-8", store=None) -> Agent:
    return Agent(
        model,
        tools=ALL_TOOLS,
        instructions=INSTRUCTIONS,
        store=store or SQLiteStore("refund_runs.db"),
        guardrails=Guardrails(allow={t.name for t in ALL_TOOLS},
                              input_guard=ticket_injection_scan),
        autonomy=Autonomy(default="autonomous", modes={"refund": "assisted"}),
    )


def args_short(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def print_inbox(pending: list) -> None:
    print("\n=== approval inbox ===")
    for p in pending:
        print(f"  action:    {p.get('tool')}({args_short(p.get('arguments', {}))})")
        print(f"  reasoning: {p.get('reasoning') or p.get('text') or '(none recorded)'}")


def print_audit(result) -> None:
    """The run, reconstructed from its event log: the audit trail."""
    print("\n=== audit trail ===")
    for e in result.events:
        when = (datetime.fromtimestamp(e.ts).strftime("%H:%M:%S")
                if e.ts else "        ")
        line = ""
        if e.type == "run_started":
            line = f"ticket: {e.payload.get('input', '')[:80]}"
        elif e.type == "model_decision":
            calls = e.payload.get("tool_calls") or []
            line = ("decided: " + ", ".join(
                f"{c['name']}({args_short(c['arguments'])})" for c in calls)
                if calls else f"concluded: {e.payload.get('text', '')[:80]}")
        elif e.type == "tool_result":
            line = (f"{e.payload.get('name', '?')} -> "
                    f"{str(e.payload.get('content', ''))[:80]}")
        elif e.type == "approval_requested":
            line = (f"PAUSED: {e.payload.get('tool')}"
                    f"({args_short(e.payload.get('arguments', {}))}) awaits a human")
        elif e.type == "approval_granted":
            line = f"APPROVED: call {e.payload.get('call_id')}"
        elif e.type == "approval_denied":
            line = (f"REJECTED: call {e.payload.get('call_id')} "
                    f"({e.payload.get('reason', '')})")
        elif e.type == "run_finished":
            line = "blocked at input" if e.payload.get("blocked") else "finished"
        print(f"  {e.seq:3d}  {when}  {e.type:19s} {line}")


async def main() -> None:
    ticket = sys.argv[1] if len(sys.argv) > 1 else (
        "I was charged twice for order ORD-88412")
    run_id = f"refund-{uuid.uuid4().hex[:8]}"
    agent = build_agent()

    result = await agent.run(ticket, run_id=run_id, trace=True)

    while result.status == "paused":
        print_inbox(result.pending_approvals)
        answer = input("\napprove? [y/N] ").strip().lower()
        if answer == "y":
            await agent.approve(run_id)
        else:
            await agent.reject(run_id, reason="declined at the console")
        result = await agent.resume(run_id, trace=True)

    print("\nFINAL:", result.output)
    print_audit(result)


if __name__ == "__main__":
    asyncio.run(main())
