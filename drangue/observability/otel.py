"""An optional OpenTelemetry tracer.

Maps drangue spans onto OpenTelemetry spans so an agent run becomes a trace like
any other distributed trace, readable in the same backend as the rest of your
system. Parent/child nesting comes from OpenTelemetry's own context.

Install the extra to use it: pip install "drangue[otel]"
"""

from __future__ import annotations

import json
import typing as t


def _attr(value: t.Any):
    # OpenTelemetry attributes must be primitives; flatten the rest to JSON.
    if isinstance(value, (str, bool, int, float)):
        return value
    return json.dumps(value, default=str)


class _OTelSpan:
    def __init__(self, otel_tracer, name: str, attrs: dict):
        self._otel_tracer = otel_tracer
        self._name = name
        self._attrs = attrs
        self._cm = None
        self._span = None

    def __enter__(self):
        self._cm = self._otel_tracer.start_as_current_span(self._name)
        self._span = self._cm.__enter__()
        for key, value in self._attrs.items():
            self.set(key, value)
        return self

    def set(self, key, value):
        # OTel rejects None attributes (logs a warning, drops them); an absent
        # value is simply not an attribute.
        if self._span is not None and value is not None:
            self._span.set_attribute(key, _attr(value))

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


class OTelTracer:
    def __init__(self, otel_tracer=None):
        if otel_tracer is None:
            try:
                from opentelemetry import trace
            except ImportError as exc:
                raise ImportError(
                    "OpenTelemetry is required for OTelTracer. "
                    "Install it with: pip install \"drangue[otel]\""
                ) from exc
            otel_tracer = trace.get_tracer("drangue")
        self._otel_tracer = otel_tracer

    def span(self, name: str, /, **attrs):
        return _OTelSpan(self._otel_tracer, name, attrs)
