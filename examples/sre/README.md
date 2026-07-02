# The SRE agent, on drangue

The companion agent to [*Production AI Agents*](https://www.llm-books.com/production-ai-agents)
implements its own orchestrator, executor, guardrails, and rollout across ~15
modules, one chapter at a time. This example is the same agent on drangue: the
six tools carry the domain, and the runtime supplies everything else — the
durable event log, the assisted-mode pause, and the audit trail.

It is also the runnable form of a governed-agent story in five beats:

1. **The incident** comes in.
2. **The agent investigates** — metrics, logs, traces, deploy ledger, runbooks.
3. **It proposes a remediation and the run pauses, durably.** The approval
   request is an event in the log; it survives a process restart and can be
   decided days later from another process.
4. **The approver sees the case, not a bare action** — the proposed call plus
   the agent's reasoning.
5. **The audit trail** replays the whole run from the log, decision included.

## Offline demo (no Docker, no API key)

```bash
PYTHONPATH=.:examples/sre python3 examples/sre/offline_demo.py
```

A scripted `FakeModel` plays the model; everything else is real — the ported
tools reading `fixtures/`, the assisted pause, the SQLite log, the audit trail.

## Live, against the book's environment

Bring up the synthetic production environment (six services, telemetry stack,
chaos engine) from the sre-agent repo, then point the agent at it:

```bash
cd ~/path/to/production-ai-agents/sre-agent && make up
export SRE_ENV=~/path/to/production-ai-agents/sre-agent
export ANTHROPIC_API_KEY=...
PYTHONPATH=.:examples/sre python3 examples/sre/sre_agent.py \
    "The orders service error rate is spiking"
```

Trigger a real incident with the chaos engine (`make chaos-day`, or inject a
single scenario) and hand the alert text to the agent. When it proposes a
remediation, the console becomes the approval inbox; approve and the agent
POSTs the service's reversible `/admin/reset`.

Endpoints default to the environment's ports and are overridable with
`PROM_URL`, `LOKI_URL`, `TEMPO_URL`, `DEPLOY_LEDGER`, `RUNBOOKS_DIR`, and
per-service URLs like `ORDERS_URL`. Set `SRE_DRY_RUN=1` to record approved
remediations without touching the environment.

## Where each production property comes from

| Property | In the book's agent | Here |
|---|---|---|
| Bounded, honest tools | hand-built `defensive_call` wrapper | `@tool(timeout=, retries=, backoff=)` |
| Forbidden actions | enforced in the tool + scope.yaml | enforced in `remediate` (the tool, not the prompt) |
| Permission scoping | `guardrails/permissions.py` | `Guardrails(allow={...})` |
| Approval pause | Postgres `approvals` table + replay | `Autonomy(modes={"remediate": "assisted"})` |
| Durability / resume | Postgres event log + checkpoints | `SQLiteStore` + replay by `run_id` |
| Audit trail | reconstructed from workflow tables | `result.events` (the log is the trail) |

The refusals worth demo-ing: ask it to restart `payments` (a known-bad
remediation, refused in code), to act on `all` services (blind restarts destroy
the diagnostic signal), or to touch a service outside the six in scope (it
escalates instead). The model can be convinced; the tool cannot.
