"""A tracer that prints a readable trace to stdout. Backs `run(trace=True)`."""

from __future__ import annotations

_DIM = "\033[2m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_RESET = "\033[0m"


def _emit(name: str, attrs: dict) -> None:
    if name == "model":
        text = attrs.get("text") or ""
        if text.strip():
            print(f"{_BLUE}*{_RESET} model  {text}")
        for c in attrs.get("tool_calls") or []:
            args = ", ".join(f"{k}={v!r}" for k, v in c["arguments"].items())
            print(f"{_GREEN}*{_RESET} tool   {c['name']}({args})")
    elif name == "tool":
        content = attrs.get("content") or ""
        preview = content if len(content) <= 200 else content[:200] + "..."
        print(f"{_DIM}       -> {preview}{_RESET}")
    elif name == "run":
        print()


class _ConsoleSpan:
    def __init__(self, name: str, attrs: dict):
        self.name = name
        self.attrs = dict(attrs)

    def set(self, key, value):
        self.attrs[key] = value

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _emit(self.name, self.attrs)
        return False


class ConsoleTracer:
    def span(self, name: str, /, **attrs):
        return _ConsoleSpan(name, attrs)
