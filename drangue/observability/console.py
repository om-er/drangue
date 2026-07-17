"""A tracer that prints a readable trace to stdout. Backs `run(trace=True)`."""

from __future__ import annotations

import os
import sys


def _codes() -> tuple[str, str, str, str]:
    """(dim, blue, green, reset), or empty strings when color would be noise.

    ANSI escapes are garbage in a redirected file or a non-VT terminal, and
    NO_COLOR (https://no-color.org) is an explicit opt-out either way.
    """
    if os.environ.get("NO_COLOR") or not getattr(sys.stdout, "isatty", lambda: False)():
        return "", "", "", ""
    return "\033[2m", "\033[34m", "\033[32m", "\033[0m"


def _emit(name: str, attrs: dict) -> None:
    dim, blue, green, reset = _codes()
    if name == "model":
        text = attrs.get("text") or ""
        if text.strip():
            print(f"{blue}*{reset} model  {text}")
        for c in attrs.get("tool_calls") or []:
            args = ", ".join(f"{k}={v!r}" for k, v in c["arguments"].items())
            print(f"{green}*{reset} tool   {c['name']}({args})")
    elif name == "tool":
        content = attrs.get("content") or ""
        preview = content if len(content) <= 200 else content[:200] + "..."
        print(f"{dim}       -> {preview}{reset}")
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
