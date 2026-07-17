"""Turn a plain typed Python function into a tool the model can call.

A tool is just a function. The decorator reads its signature and docstring
and builds the JSON schema for you. No manual schema, no Pydantic required.
"""

from __future__ import annotations

import inspect
import types
import typing as t
from dataclasses import dataclass, replace

from .hardening import DEFAULT_BACKOFF, ToolPolicy, _UNSET, make_policy

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
        if annotation is type(None):
            return {"type": "null"}
        return {"type": _PY_TO_JSON.get(annotation, "string")}

    if origin in (list, tuple):
        args = t.get_args(annotation)
        item = _json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}

    if origin is dict:
        return {"type": "object"}

    if origin is t.Literal:
        return {"enum": list(t.get_args(annotation))}

    # typing.Union covers Optional[X] / Union[...]; types.UnionType is PEP 604
    # (`int | None`), which get_origin reports as a distinct origin.
    if origin in (t.Union, types.UnionType):
        schemas = [_json_type(a) for a in t.get_args(annotation)]
        if all(set(s) == {"type"} for s in schemas):
            merged: list[str] = []
            for s in schemas:
                if s["type"] not in merged:
                    merged.append(s["type"])
            return {"type": merged[0] if len(merged) == 1 else merged}
        return {"anyOf": schemas}

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
    reversible: bool = True            # a wrong call is recoverable
    requires_approval: bool = False    # must pass an action gate before running

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
         retries: int = 0, backoff: float = DEFAULT_BACKOFF,
         max_backoff: float = 30.0,
         retry_on=None, validate=None, fallback=_UNSET,
         reversible: bool = True, requires_approval: bool = False):
    """Decorate a function to expose it as a tool.

    Usage:
        @tool
        def get_weather(city: str) -> str:
            "Get the current weather for a city."
            ...

    The function name becomes the tool name and the docstring becomes the
    description, unless you override them.

    Hardening options (Chapter 6), all optional:
        timeout     seconds before the call is abandoned (not retried by
                    default: a timed-out sync tool may still be running)
        retries     extra attempts after the first, for transient failures
        backoff     base seconds for jittered exponential backoff between retries
        retry_on    exception types to treat as retryable; REPLACES the default
                    set (drangue.hardening.DEFAULT_RETRY_ON), so it can narrow
                    as well as widen what is retried
        validate    a callable run on the result; raise ValidationError to
                    reject, return a replacement to transform, or return None
                    to keep the result as-is
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
            reversible=reversible,
            requires_approval=requires_approval,
        )

    if func is not None:
        return wrap(func)
    return wrap


def harden(obj, **policy_kwargs) -> Tool:
    """Return a copy of a tool (or function) with a hardening policy attached.

        flaky = harden(flaky, timeout=2.0, retries=3, backoff=0.5)

    The original is left untouched, so hardening a tool shared by two agents
    never silently changes the other agent's policy.
    """
    t_obj = obj if isinstance(obj, Tool) else tool(obj)
    return replace(t_obj, policy=make_policy(**policy_kwargs))
