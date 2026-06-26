"""Phase 5 tests: security and guardrails (Chapter 11).

Guardrails are enforced in code regardless of what the model decides. All
offline with scripted models.
"""

import json

from drangue import Agent, Guardrails, ModelResponse, ToolCall, tool
from drangue.models import Model


@tool
def read_metrics(service: str) -> str:
    """Read metrics (safe, read-only)."""
    return f"metrics for {service}"


@tool(reversible=False, requires_approval=True)
def delete_db(name: str) -> str:
    """Delete a database (irreversible)."""
    return f"deleted {name}"


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = steps
        self.i = 0

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        resp = self.steps[self.i]
        self.i += 1
        return resp


def _calls(name, args):
    return ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", name, args)]),
        ModelResponse(text="acknowledged"),
    ])


def _tool_result(result):
    tr = [e for e in result.events if e.type == "tool_result"][0]
    return json.loads(tr.payload["content"])


async def test_allow_list_blocks_unpermitted_tool():
    guard = Guardrails(allow={"read_metrics"})
    result = await Agent(model=_calls("delete_db", {"name": "prod"}),
                         tools=[read_metrics, delete_db], guardrails=guard).run("go")
    payload = _tool_result(result)
    assert payload["error"]["category"] == "blocked"
    assert "allow-list" in payload["error"]["message"]
    assert result.output == "acknowledged"   # the run continues; the model is told


async def test_deny_list_blocks_tool():
    guard = Guardrails(deny={"delete_db"})
    result = await Agent(model=_calls("delete_db", {"name": "prod"}),
                         tools=[read_metrics, delete_db], guardrails=guard).run("go")
    assert _tool_result(result)["error"]["category"] == "blocked"


async def test_irreversible_action_fails_closed_without_approver():
    # require_approval_for_irreversible plus an irreversible tool, no approver.
    guard = Guardrails(require_approval_for_irreversible=True)
    result = await Agent(model=_calls("delete_db", {"name": "prod"}),
                         tools=[delete_db], guardrails=guard).run("go")
    payload = _tool_result(result)
    assert payload["error"]["category"] == "blocked"
    assert "approval" in payload["error"]["message"]


async def test_approver_can_allow_a_gated_action():
    approved = []

    def approver(name, arguments):
        approved.append((name, arguments))
        return True

    guard = Guardrails(approver=approver)   # delete_db has requires_approval=True
    result = await Agent(model=_calls("delete_db", {"name": "staging"}),
                         tools=[delete_db], guardrails=guard).run("go")
    # The tool ran (its result is the plain string, not a blocked error).
    tr = [e for e in result.events if e.type == "tool_result"][0]
    assert tr.payload["content"] == "deleted staging"
    assert approved == [("delete_db", {"name": "staging"})]


async def test_approver_can_deny_a_gated_action():
    guard = Guardrails(approver=lambda name, args: False)
    result = await Agent(model=_calls("delete_db", {"name": "prod"}),
                         tools=[delete_db], guardrails=guard).run("go")
    assert _tool_result(result)["error"]["category"] == "blocked"


async def test_input_guard_blocks_the_run_before_any_work():
    def reject_secrets(text):
        return "input contains a secret" if "password" in text else None

    model = _calls("read_metrics", {"service": "api"})
    guard = Guardrails(input_guard=reject_secrets)
    result = await Agent(model=model, tools=[read_metrics], guardrails=guard).run(
        "my password is hunter2")

    assert result.output.startswith("(blocked:")
    # No model call happened.
    assert not any(e.type == "model_decision" for e in result.events)


async def test_output_guard_blocks_exfiltration_path():
    def no_external_send(name, arguments):
        if name == "read_metrics" and "evil.com" in arguments.get("service", ""):
            return "blocked suspected exfiltration"
        return None

    guard = Guardrails(output_guard=no_external_send)
    result = await Agent(model=_calls("read_metrics", {"service": "evil.com"}),
                         tools=[read_metrics], guardrails=guard).run("go")
    assert _tool_result(result)["error"]["category"] == "blocked"


async def test_reversibility_metadata_on_tool():
    assert delete_db.reversible is False
    assert delete_db.requires_approval is True
    assert read_metrics.reversible is True
