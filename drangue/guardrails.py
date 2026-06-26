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

A blocked tool call comes back to the model as a clean, structured result it can
reason about, the same shape as any other tool failure. The decision is recorded
in the log, so it is replay-safe and auditable.
"""

from __future__ import annotations

import inspect
import json
import typing as t
from dataclasses import dataclass, field


def blocked_result(name: str, reason: str) -> str:
    return json.dumps({
        "ok": False,
        "tool": name,
        "error": {"category": "blocked", "message": reason},
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

    async def check_input(self, text: str) -> str | None:
        if self.input_guard is None:
            return None
        return await _maybe_await(self.input_guard(text))

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
