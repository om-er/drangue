"""The book's SRE agent, on drangue: investigate, propose, pause for approval.

The companion agent to *Production AI Agents* implements its own orchestrator,
executor, guardrails, and rollout across ~15 modules. On drangue the same
architecture is this one constructor: the tools carry the domain, and the
runtime supplies durability, the approval pause, and the audit trail.

Live use needs the book's synthetic environment running (`make up` in the
sre-agent repo) and an ANTHROPIC_API_KEY. For a self-contained run with
neither, see offline_demo.py.

    export SRE_ENV=~/Documents/books/my-books/production-ai-agents/sre-agent
    python examples/sre/sre_agent.py "Orders error rate is spiking"
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime

from drangue import Agent, Autonomy, Guardrails, SQLiteStore

from sre_tools import ALL_TOOLS

INSTRUCTIONS = """You are an on-call SRE agent for a six-service e-commerce
checkout: web, api-gateway, orders, payments, inventory, notifications.
Anything else is out of scope: say so and escalate instead of acting.

Investigate evidence-first: correlate metrics, logs, and deploy history before
diagnosing. Consult runbooks for known failure modes. State the diagnosis with
its evidence, then — only when the evidence supports it — propose a remediation
with the remediate tool. Remediations are approval-gated; make the case for the
action in your reasoning, because that is what the approver reads. Be concise
and specific."""


def build_agent(model="claude-opus-4-8", store=None) -> Agent:
    return Agent(
        model,
        tools=ALL_TOOLS,
        instructions=INSTRUCTIONS,
        store=store or SQLiteStore("sre_runs.db"),
        guardrails=Guardrails(allow={t.name for t in ALL_TOOLS}),
        autonomy=Autonomy(default="autonomous", modes={"remediate": "assisted"}),
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
            line = f"incident: {e.payload.get('input', '')[:80]}"
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
            line = "finished"
        print(f"  {e.seq:3d}  {when}  {e.type:19s} {line}")


async def main() -> None:
    incident = sys.argv[1] if len(sys.argv) > 1 else (
        "The orders service error rate is spiking and checkout latency is up.")
    run_id = f"incident-{uuid.uuid4().hex[:8]}"
    agent = build_agent()

    result = await agent.run(incident, run_id=run_id, trace=True)

    # An assisted action is a durable pause: the request is an event in the
    # log, so this loop could equally run days later, in a new process.
    while result.status == "paused":
        print_inbox(result.pending_approvals)
        answer = input("\napprove? [y/N] ").strip().lower()
        if answer == "y":
            await agent.approve(run_id)
        else:
            await agent.reject(run_id, reason="operator declined at the console")
        result = await agent.resume(run_id, trace=True)

    print("\nFINAL:", result.output)
    print_audit(result)


if __name__ == "__main__":
    asyncio.run(main())
