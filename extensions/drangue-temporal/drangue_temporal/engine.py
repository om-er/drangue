"""Run drangue agents on Temporal.

Design: Temporal supervises our durable engine rather than replacing it. A
workflow drives one run; the actual work happens in an activity that rebuilds the
agent from a registry (via the RunSpec seam) and runs the in-process
`EventSourcedEngine` against the agent's shared durable store (e.g. Postgres).

Why this shape:
- It reuses the whole tested engine, so budget, autonomy, memory, and guardrails
  all work unchanged. A workflow that reimplemented the loop would lose them.
- The activity is safe to retry: a retry resumes from the store by run_id rather
  than redoing recorded steps. So Temporal's run-level retry composes with our
  step-level durability.
- Human-in-the-loop maps onto Temporal's durable waits: when a run pauses for
  approval, the workflow waits (possibly for days) on an `approve` signal, records
  the approval, and re-runs the activity, which resumes past the gate.

The worker must register how to rebuild each agent, and those agents must point
at a shared durable store so resume works across workers:

    from drangue import Agent, RunSpec, SQLiteStore
    from drangue_temporal import REGISTRY, build_worker, submit_run, approve

    REGISTRY.register("sre", lambda: Agent(
        "claude-opus-4-8", tools=TOOLS,
        store=SQLiteStore("/shared/runs.db"),   # a real shared DB in production
    ))
    worker = build_worker(client, task_queue="drangue")   # then worker.run()
    await submit_run(client, RunSpec("sre", "inc-1", "investigate"), task_queue="drangue")
    await approve(client, "inc-1")                          # when a human approves
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

from drangue import AgentRegistry, EventSourcedEngine, RunSpec, context_from_spec

# The worker populates this; activities resolve agents from it. It is module-level
# because activities cannot receive live objects, only serializable arguments.
REGISTRY = AgentRegistry()


@dataclass
class RunOutcome:
    run_id: str
    status: str        # completed | paused
    output: str


@activity.defn
async def execute_run(spec_dict: dict) -> dict:
    """Run or resume an agent to its next stopping point, on a worker."""
    spec = RunSpec.from_dict(spec_dict)
    ctx = context_from_spec(spec, REGISTRY)
    result = await EventSourcedEngine().run(ctx)
    return {"run_id": spec.run_id, "status": result.status, "output": result.output}


@activity.defn
async def record_approval(spec_dict: dict) -> None:
    """Approve the pending action so the next execute_run resumes past the gate."""
    spec = RunSpec.from_dict(spec_dict)
    agent = REGISTRY.build(spec.key)
    await agent.approve(spec.run_id)


@workflow.defn
class AgentRun:
    def __init__(self):
        self._approved = False

    @workflow.signal
    async def approve(self) -> None:
        self._approved = True

    @workflow.run
    async def run(self, spec_dict: dict) -> dict:
        # The workflow body is trivially deterministic: it only schedules
        # activities and waits on signals. All non-determinism lives in the
        # activity, where it belongs.
        while True:
            outcome = await workflow.execute_activity(
                execute_run, spec_dict,
                start_to_close_timeout=timedelta(minutes=10),
                # Unlimited attempts: a crashed activity resumes from the store.
                retry_policy=RetryPolicy(maximum_attempts=0),
            )
            if outcome["status"] != "paused":
                return outcome

            # Durably wait for a human, then record the approval and resume.
            await workflow.wait_condition(lambda: self._approved)
            self._approved = False
            await workflow.execute_activity(
                record_approval, spec_dict,
                start_to_close_timeout=timedelta(seconds=30),
            )


def build_worker(client, *, task_queue: str):
    """Build a Temporal worker that serves drangue runs on `task_queue`."""
    from temporalio.worker import Worker
    return Worker(
        client, task_queue=task_queue,
        workflows=[AgentRun],
        activities=[execute_run, record_approval],
    )


async def submit_run(client, spec: RunSpec, *, task_queue: str):
    """Start a workflow for a run. Returns the Temporal workflow handle."""
    return await client.start_workflow(
        AgentRun.run, spec.to_dict(),
        id=f"drangue-{spec.run_id}", task_queue=task_queue,
    )


async def approve(client, run_id: str) -> None:
    """Signal approval to a paused run."""
    handle = client.get_workflow_handle(f"drangue-{run_id}")
    await handle.signal(AgentRun.approve)
