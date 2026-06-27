"""Agent registry: the worker-side construction boundary for distributed engines.

A distributed engine (Temporal, DBOS) ships a serializable RunSpec across the
wire. The worker holds an AgentRegistry mapping the spec's key to a factory that
rebuilds a live Agent locally (its model, tools, guardrails, memory). The engine
then turns the spec into a RunContext with `context_from_spec` and drives the
same loop the in-process engine does.

This is the piece that was missing for distributed engines: the orchestrator's
determinism rules already line up with workflow engines; the only gap was a
serializable way to reconstruct the agent on the worker. In-process code does not
use any of this.
"""

from __future__ import annotations

import typing as t


class AgentRegistry:
    """Maps a serializable key to a factory that builds a live Agent."""

    def __init__(self):
        self._factories: dict[str, t.Callable] = {}

    def register(self, key: str, factory: t.Callable) -> None:
        """Register how to build the agent named `key` on this worker."""
        self._factories[key] = factory

    def build(self, key: str):
        factory = self._factories.get(key)
        if factory is None:
            raise KeyError(f"no agent registered under {key!r}")
        return factory()


def context_from_spec(spec, registry: AgentRegistry, *, emit=None,
                      trace: bool = False):
    """Rebuild a live RunContext from a serializable RunSpec via the registry.

    This is the worker-side construction a distributed engine performs after a
    spec crosses the boundary. The rebuilt agent must point at the shared durable
    store (a real DB) for resume to work across workers.
    """
    agent = registry.build(spec.key)
    return agent._context(spec.input, spec.run_id, trace=trace, emit=emit)
