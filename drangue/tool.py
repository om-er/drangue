"""Turn a plain typed Python function into a tool the model can call.

A tool is just a function. The decorator reads its signature and docstring
and builds the JSON schema for you. No manual schema, no Pydantic required.
"""

from __future__ import annotations

import inspect
import typing as t
from dataclasses import dataclass

from .hardening import ToolPolicy, _UNSET, make_policy

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: t.Any) -> dict:
    """Map a Python type annotation to a JSON Schema fragment."""
    origin = t.get_origin(annotation)

    if origin is None:
        return {"type": _PY_TO_JSON.get(annotation, "string")}

    if origin in (list, tuple):
        args = t.get_args(annotation)
        item = _json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}

    if origin is dict:
        return {"type": "object"}

    if origin is t.Union:
        # Optional[X] / Union[X, None]: use the first non-None member.
        members = [a for a in t.get_args(annotation) if a is not type(None)]
        return _json_type(members[0]) if members else {"type": "string"}

    return {"type": "string"}


# Parameters the runtime injects, hidden from the model-facing schema.
RESERVED_PARAMS = {"idempotency_key"}


@dataclass
class Tool:
    """A callable plus the schema describing how a model should call it."""

    name: str
    description: str
    parameters: dict
    func: t.Callable
    wants_idempotency_key: bool = False
    policy: ToolPolicy | None = None

    def to_schema(self) -> dict:
        """Return the tool definition in the shape the model API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def __call__(self, **kwargs):
        return self.func(**kwargs)


def tool(func: t.Callable | None = None, *, name: str | None = None,
         description: str | None = None, timeout: float | None = None,
         retries: int = 0, backoff: float = 0.0, max_backoff: float = 30.0,
         retry_on=None, validate=None, fallback=_UNSET):
    """Decorate a function to expose it as a tool.

    Usage:
        @tool
        def get_weather(city: str) -> str:
            "Get the current weather for a city."
            ...

    The function name becomes the tool name and the docstring becomes the
    description, unless you override them.

    Hardening options (Chapter 6), all optional:
        timeout     seconds before the call is abandoned (a retryable timeout)
        retries     extra attempts after the first, for transient failures
        backoff     base seconds for exponential backoff between retries
        retry_on    extra exception types to treat as transient
        validate    a callable run on the result; raise ValidationError to reject
        fallback    a value (or callable) returned, marked degraded, on failure
    """

    policy = make_policy(timeout=timeout, retries=retries, backoff=backoff,
                         max_backoff=max_backoff, retry_on=retry_on,
                         validate=validate, fallback=fallback)

    def wrap(fn: t.Callable) -> Tool:
        sig = inspect.signature(fn)
        try:
            hints = t.get_type_hints(fn)
        except Exception:
            hints = {}

        properties: dict = {}
        required: list[str] = []

        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if pname in RESERVED_PARAMS:
                continue  # runtime-injected; never shown to or filled by the model
            properties[pname] = _json_type(hints.get(pname, str))
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        parameters: dict = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required

        return Tool(
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn) or "",
            parameters=parameters,
            func=fn,
            wants_idempotency_key="idempotency_key" in sig.parameters,
            policy=policy,
        )

    if func is not None:
        return wrap(func)
    return wrap


def harden(obj, **policy_kwargs) -> Tool:
    """Attach (or replace) a hardening policy on an existing tool or function.

        flaky = harden(flaky, timeout=2.0, retries=3, backoff=0.5)
    """
    t_obj = obj if isinstance(obj, Tool) else tool(obj)
    t_obj.policy = make_policy(**policy_kwargs)
    return t_obj
