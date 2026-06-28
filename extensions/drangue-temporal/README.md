# drangue-temporal

Run drangue agents on [Temporal](https://temporal.io). Implements the Engine seam
by **supervising** the durable engine rather than replacing it: a workflow drives
a run; the work happens in an activity that rebuilds the agent (via the `RunSpec`
construction seam) and runs the in-process `EventSourcedEngine` against a shared
durable store.

## Why this shape

- **Feature-complete by reuse.** The activity runs the full engine, so budget,
  autonomy, memory, and guardrails all work unchanged. A workflow that
  reimplemented the loop would lose them.
- **Retry composes with resume.** The activity is safe to retry: a retry resumes
  from the store by `run_id` instead of redoing recorded steps. Temporal's
  run-level retry/visibility sits on top of the engine's step-level durability.
- **Human-in-the-loop is a durable wait.** When a run pauses for approval, the
  workflow waits (possibly for days) on an `approve` signal, records the approval,
  and re-runs the activity, which resumes past the gate.

## Usage

```python
from drangue import Agent, RunSpec, SQLiteStore
from drangue_temporal import REGISTRY, build_worker, submit_run, approve

# On the worker: how to rebuild each agent. Point it at a SHARED durable store
# (a real DB in production) so resume works across workers.
REGISTRY.register("sre", lambda: Agent(
    "claude-opus-4-8", tools=TOOLS, store=SQLiteStore("/shared/runs.db")))

worker = build_worker(client, task_queue="drangue")   # then await worker.run()

# Elsewhere: start a run, and approve it when a human signs off.
await submit_run(client, RunSpec("sre", "inc-1", "investigate orders latency"),
                 task_queue="drangue")
await approve(client, "inc-1")
```

The workflow body is trivially deterministic (it only schedules activities and
waits on a signal); all non-determinism lives in the activity, where it belongs.
That lines up with the orchestrator's own determinism rules, which is why this
fits Temporal cleanly.

## Status: verified

Both paths are tested against Temporal's time-skipping test server
(`tests/test_temporal.py`): a run-to-completion, and a run that pauses on an
assisted action, waits for the `approve` signal, and resumes to completion.

```bash
pip install temporalio
DRANGUE_TEMPORAL_TEST=1 python extensions/run_tests.py   # from the repo root
```

Without `temporalio` and the flag, the suite skips cleanly.
