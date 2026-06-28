"""drangue-temporal tests.

Requires the temporalio SDK and its time-skipping test server. Set
DRANGUE_TEMPORAL_TEST=1 to run (the test server is fetched on first use);
otherwise this skips. Imports live inside the test so the module loads, and
skips cleanly, without temporalio installed.
"""

import os

RUN = os.environ.get("DRANGUE_TEMPORAL_TEST")


async def test_workflow_runs_and_resumes_an_agent():
    if not RUN:
        print("SKIP test_workflow_runs_and_resumes_an_agent: "
              "set DRANGUE_TEMPORAL_TEST=1 (needs temporalio + test server)")
        return

    import tempfile

    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from drangue import Agent, RunSpec, SQLiteStore
    from drangue.testing import FakeModel
    from drangue_temporal import (
        REGISTRY,
        AgentRun,
        execute_run,
        record_approval,
    )

    path = os.path.join(tempfile.mkdtemp(), "temporal.db")
    REGISTRY.register("greeter", lambda: Agent(
        model=FakeModel(final="hi from temporal"), tools=[], store=SQLiteStore(path)))

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(env.client, task_queue="tq", workflows=[AgentRun],
                          activities=[execute_run, record_approval]):
            handle = await env.client.start_workflow(
                AgentRun.run, RunSpec("greeter", "job-1", "hi").to_dict(),
                id="drangue-job-1", task_queue="tq")
            result = await handle.result()

    assert result["status"] == "completed"
    assert result["output"] == "hi from temporal"


async def test_workflow_pauses_for_approval_then_resumes():
    if not RUN:
        print("SKIP test_workflow_pauses_for_approval_then_resumes: "
              "set DRANGUE_TEMPORAL_TEST=1")
        return

    import tempfile

    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from drangue import (
        Agent,
        Autonomy,
        ModelResponse,
        RunSpec,
        SQLiteStore,
        ToolCall,
        tool,
    )
    from drangue.models import Model
    from drangue_temporal import REGISTRY, AgentRun, execute_run, record_approval

    path = os.path.join(tempfile.mkdtemp(), "hitl.db")
    ran = []

    @tool(reversible=False)
    def remediate(x: int) -> str:
        """Assisted remediation."""
        ran.append("remediate")
        return "fixed"

    class Reactive(Model):
        async def generate(self, *, system, messages, tools, idempotency_key=None):
            used = [m["name"] for m in messages if m.get("role") == "tool"]
            if "remediate" not in used:
                return ModelResponse(
                    tool_calls=[ToolCall("c1", "remediate", {"x": 1})],
                    reasoning="metrics point to a fix")
            return ModelResponse(text="resolved")

    REGISTRY.register("sre", lambda: Agent(
        model=Reactive(), tools=[remediate], store=SQLiteStore(path),
        autonomy=Autonomy(modes={"remediate": "assisted"})))

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(env.client, task_queue="tq2", workflows=[AgentRun],
                          activities=[execute_run, record_approval]):
            handle = await env.client.start_workflow(
                AgentRun.run, RunSpec("sre", "inc-9", "investigate").to_dict(),
                id="drangue-inc-9", task_queue="tq2")
            # The run pauses on the assisted action; approve so it resumes.
            await handle.signal(AgentRun.approve)
            result = await handle.result()

    assert result["status"] == "completed"
    assert result["output"] == "resolved"
    assert ran == ["remediate"]      # the gated tool ran exactly once, after approval
