"""Structured output: schema-constrained final answers, provider-neutral.

When a run has an `output_schema`, a synthetic `final_answer` tool built from
that schema is shown to the model; calling it delivers the answer. The engine
validates the arguments (never executes anything), records the accepted
answer, and ends the run. A plain-text finish is accepted if the text parses
and validates as the schema, corrected with recorded feedback otherwise, and
after `MAX_SCHEMA_FEEDBACK` failed corrections the run finishes gracefully
with `structured: false` rather than looping forever.

The validator is a deliberate zero-dependency subset of JSON Schema: `type`
(including type lists), `enum`, `required`, `properties`, and `items`. It is
pure, so acceptance is replay-deterministic. For anything richer, validate
downstream from `result.output_parsed`.
"""

from __future__ import annotations

import json

from .tool import Tool

FINAL_ANSWER = "final_answer"
MAX_SCHEMA_FEEDBACK = 2

_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def validate(schema: dict, value, path: str = "$") -> list[str]:
    """Errors for `value` against the supported JSON Schema subset ([] = ok)."""
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        py = tuple(_TYPES[t] for t in types if t in _TYPES)
        ok = isinstance(value, py) if py else True
        # bool is an int subclass; do not let True satisfy "integer".
        if ok and isinstance(value, bool) and "boolean" not in types:
            ok = False
        if not ok:
            errors.append(f"{path}: expected {expected}, got {type(value).__name__}")
            return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        for name, sub in (schema.get("properties") or {}).items():
            if name in value:
                errors.extend(validate(sub, value[name], f"{path}.{name}"))
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate(schema["items"], item, f"{path}[{i}]"))
    return errors


def final_answer_tool(schema: dict) -> Tool:
    """The synthetic tool a schema-constrained run shows to the model.

    It exists only as a schema: the engine intercepts calls to it and never
    dispatches, so the no-op func is unreachable by construction.
    """
    return Tool(
        name=FINAL_ANSWER,
        description=("Deliver your final answer. Call this exactly once, when "
                     "you are done, with arguments matching the schema; the "
                     "run ends with what you pass here."),
        parameters=schema,
        func=lambda **kw: "",
    )


def parse_text_answer(text: str, schema: dict):
    """(parsed, errors) for a plain-text finish under a schema."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, ["the final answer was not valid JSON"]
    errors = validate(schema, parsed)
    return (parsed, errors) if not errors else (None, errors)


def feedback_text(errors: list[str]) -> str:
    return ("Your final answer did not match the required schema: "
            + "; ".join(errors[:5])
            + ". Call the final_answer tool with arguments that satisfy the "
              "schema.")
