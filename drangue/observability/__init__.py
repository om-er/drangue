"""The tracer seam: live export of a run's spans.

Two ways to see a run:

  - `Result.trace` reconstructs a span tree from the persisted log. Always
    available, always faithful, no setup. Best for tests and after-the-fact.
  - A `Tracer` receives spans as they happen, for live or external export.
    `NullTracer` (default) does nothing, `ConsoleTracer` prints, `OTelTracer`
    emits OpenTelemetry spans into your existing tracing backend.

A tracer just needs `span(name, **attrs)` returning a context manager whose
`set(key, value)` attaches attributes. The executor opens a span per step; the
engine opens the root run span. Nesting follows the call stack.
"""

from __future__ import annotations

import typing as t

from ..events import Span


class Tracer(t.Protocol):
    def span(self, name: str, /, **attrs): ...


class _NullSpan:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set(self, key, value):
        pass


class NullTracer:
    """Does nothing. The default."""

    def span(self, name: str, /, **attrs):
        return _NullSpan()


from .console import ConsoleTracer  # noqa: E402
from .otel import OTelTracer  # noqa: E402

__all__ = ["Tracer", "Span", "NullTracer", "ConsoleTracer", "OTelTracer"]
