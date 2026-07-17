"""Security guardrails (Chapter 11).

Defense in depth: assume prompt injection will sometimes succeed, and make that
success survivable. The layers here are deterministic and enforced in code, not
in the prompt, because a prompt instruction is exactly what injection overrides.

  1. Input guardrail   - inspect the run's input before work starts.
  2. Permission scoping - an allow/deny list of tools the agent may call. The
     agent is a constrained principal, not the user.
  3. Action gates      - consequential or irreversible actions require approval.
  4. Output guardrail  - inspect a proposed tool call (its name and arguments)
     before it runs, e.g. to block an exfiltration path.
  5. Result guardrail  - inspect a tool's RESULT before the model sees it.
     Tool results are the classic injection carrier (a fetched web page, a
     ticket body); a flagged result is withheld from the conversation.

A blocked tool call comes back to the model as a clean, structured result it can
reason about, the same shape as any other tool failure. The block itself is
recorded in the log (as the tool_result content), so it is replay-safe and
auditable; a custom approver's decisions are yours to persist if you need an
audit trail beyond the block.
"""

from __future__ import annotations

import inspect
import json
import typing as t
from dataclasses import dataclass, field


def blocked_result(name: str, reason: str, *, category: str = "blocked") -> str:
    return json.dumps({
        "ok": False,
        "tool": name,
        "error": {"category": category, "message": reason},
    })


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass
class Guardrails:
    allow: set | None = None                 # tool allow-list; None means all configured tools
    deny: set = field(default_factory=set)    # tool deny-list
    require_approval_for_irreversible: bool = False
    approver: t.Callable | None = None        # (tool_name, arguments) -> bool
    input_guard: t.Callable | None = None     # (text) -> reason | None
    output_guard: t.Callable | None = None    # (tool_name, arguments) -> reason | None
    result_guard: t.Callable | None = None    # (tool_name, result_text) -> reason | None

    async def check_input(self, text: str) -> str | None:
        if self.input_guard is None:
            return None
        return await _maybe_await(self.input_guard(text))

    async def check_result(self, tool_name: str, content: str) -> str | None:
        """Screen a tool's result before it enters the conversation.

        A non-None reason withholds the result: the model receives a clean
        `result_blocked` failure instead of the flagged content. The side
        effect (if any) has already happened — this layer protects the
        conversation, not the world. For redaction rather than blocking, wrap
        the tool itself (e.g. a `validate` transform in its policy).
        """
        if self.result_guard is None:
            return None
        return await _maybe_await(self.result_guard(tool_name, content))

    async def check_tool(self, tool, arguments: dict) -> str | None:
        name = tool.name

        # Permission scoping.
        if self.allow is not None and name not in self.allow:
            return f"tool '{name}' is not in the allow-list"
        if name in self.deny:
            return f"tool '{name}' is denied"

        # Output guardrail on the proposed call.
        if self.output_guard is not None:
            reason = await _maybe_await(self.output_guard(name, arguments))
            if reason:
                return reason

        # Action gate.
        needs_approval = tool.requires_approval or (
            self.require_approval_for_irreversible and not tool.reversible
        )
        if needs_approval:
            if self.approver is None:
                return f"tool '{name}' requires approval but no approver is configured"
            approved = await _maybe_await(self.approver(name, arguments))
            if not approved:
                return f"tool '{name}' was not approved"

        return None
