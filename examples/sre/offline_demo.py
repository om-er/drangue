"""The SRE approval flow, fully offline: no Docker, no API key.

A scripted FakeModel plays the model's part with a plausible investigation, so
what this demo actually exercises is everything else — the real ported tools
(deploy ledger and runbooks from fixtures/), the real assisted-mode pause, a
real SQLite event log, and the audit trail reconstructed from it.

    python examples/sre/offline_demo.py

The five beats of the governed-agent story, in order:
  1. the incident comes in,
  2. the agent investigates with tools,
  3. it proposes a remediation and the run PAUSES, durably,
  4. the approver sees the action plus the agent's reasoning and decides,
  5. the audit trail shows the whole thing, decision included.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

os.environ["SRE_DRY_RUN"] = "1"  # remediation is recorded, not applied

from drangue import SQLiteStore
from drangue.models import ModelResponse, ToolCall
from drangue.testing import FakeModel

from sre_agent import build_agent, print_audit, print_inbox

INCIDENT = "The orders service error rate is spiking and checkout latency is up."

# The scripted investigation. Each response is one model turn; the reasoning
# strings are what the approval inbox will show the human.
SCRIPT = [
    ModelResponse(
        reasoning="A new error spike is most often a recent deploy; check the ledger first.",
        tool_calls=[ToolCall(id="c1", name="deploy_history",
                             arguments={"service": "orders"})],
    ),
    ModelResponse(
        reasoning="orders v2026.07.02.1 shipped ~1h before the spike; look for a known failure mode.",
        tool_calls=[ToolCall(id="c2", name="runbook_search",
                             arguments={"query": "orders error spike deploy"})],
    ),
    ModelResponse(
        reasoning=("Diagnosis: the 08:05 orders deploy dropped the order-lookup index "
                   "(runbook: orders-error-spike). The documented fix is a reversible "
                   "restart of orders, which restores the degraded state. Low blast "
                   "radius, safe to repeat. Requesting approval to restart orders."),
        tool_calls=[ToolCall(id="c3", name="remediate",
                             arguments={"service": "orders", "action": "restart"})],
    ),
    ModelResponse(
        text=("Incident resolved. Root cause: the v2026.07.02.1 orders deploy dropped "
              "the order-lookup index, turning point reads into scans (per the "
              "orders-error-spike runbook). Remediation: approved restart of the "
              "orders service. Recommend watching the 5xx rate for 15 minutes."),
    ),
]


async def main() -> None:
    db = Path(tempfile.mkdtemp()) / "sre_demo.db"
    agent = build_agent(model=FakeModel(SCRIPT), store=SQLiteStore(str(db)))
    run_id = "incident-demo-1"

    print(f"incident: {INCIDENT}")
    result = await agent.run(INCIDENT, run_id=run_id)

    assert result.status == "paused", f"expected a paused run, got {result.status}"
    print_inbox(result.pending_approvals)

    # The pause is durable: the request is an event in the SQLite log, so the
    # decision could come from another process, days later. Here we approve.
    print("\napprove? [y/N] y   (scripted)")
    await agent.approve(run_id)
    result = await agent.resume(run_id)

    print("\nFINAL:", result.output)
    print_audit(result)
    print(f"\n(event log persisted at {db} — rerun with the same run_id and "
          f"the recorded steps replay instead of re-executing)")


if __name__ == "__main__":
    asyncio.run(main())
