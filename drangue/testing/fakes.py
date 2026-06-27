"""Canonical offline fakes for building tests without real services.

These are the same shapes used throughout drangue's own test suite, published so
battery authors can construct agents and assert behavior without a live model,
store, or tracing backend.
"""

from __future__ import annotations

from ..models import Model, ModelResponse


class FakeModel(Model):
    """A deterministic, offline model.

    Pass `responses` (a list of ModelResponse) to script a run turn by turn, or
    rely on `final` to answer with fixed text. Counts calls for assertions.
    """

    def __init__(self, responses=None, *, final: str = "done"):
        self.responses = list(responses) if responses else None
        self.final = final
        self.calls = 0
        self.model = "fake"

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        self.calls += 1
        if self.responses:
            i = min(self.calls - 1, len(self.responses) - 1)
            return self.responses[i]
        return ModelResponse(text=self.final)


class RecordingTracer:
    """A tracer that collects (name, attrs) per closed span, for assertions."""

    def __init__(self):
        self.spans: list = []

    def span(self, name: str, /, **attrs):
        return _RecordingSpan(self, name, attrs)


class _RecordingSpan:
    def __init__(self, tracer: RecordingTracer, name: str, attrs: dict):
        self._tracer = tracer
        self.name = name
        self.attrs = dict(attrs)

    def set(self, key, value):
        self.attrs[key] = value

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._tracer.spans.append((self.name, self.attrs))
        return False
