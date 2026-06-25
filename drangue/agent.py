"""The agent: a model, some tools, and a readable loop.

`agent.run(prompt)` is the whole happy path. It is a plain while-loop you
could have written yourself: ask the model, run any tools it asked for, feed
the results back, repeat until it stops asking.
"""

from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass

from .models import AnthropicModel, Model, ToolCall, ToolResult
from .tool import Tool
from .tool import tool as make_tool

_DIM = "\033[2m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_RESET = "\033[0m"


@dataclass
class Event:
    """One thing that happened during a run. Yielded by `stream`."""

    type: str  # "model" | "tool_call" | "tool_result" | "final"
    text: str = ""
    name: str = ""
    arguments: dict | None = None
    result: str = ""


@dataclass
class Result:
    """The outcome of a run."""

    output: str
    messages: list
    steps: int


def _resolve_model(model: t.Any, max_tokens: int) -> Model:
    if isinstance(model, str):
        return AnthropicModel(model, max_tokens=max_tokens)
    return model  # assume it implements the Model interface


def _as_tool(obj: t.Any) -> Tool:
    if isinstance(obj, Tool):
        return obj
    if callable(obj):
        return make_tool(obj)
    raise TypeError(f"Expected a tool or callable, got {type(obj)!r}")


def _print_model(text: str) -> None:
    if text.strip():
        print(f"{_BLUE}*{_RESET} model  {text}")


def _print_tool_call(call: ToolCall) -> None:
    args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    print(f"{_GREEN}*{_RESET} tool   {call.name}({args})")


def _print_tool_result(result: str) -> None:
    preview = result if len(result) <= 200 else result[:200] + "..."
    print(f"{_DIM}       -> {preview}{_RESET}")


class Agent:
    """A model plus tools. Call `run` or `stream`."""

    def __init__(self, model: t.Any, tools: list | None = None,
                 instructions: str = "", *, max_steps: int = 20,
                 max_tokens: int = 4096):
        self.model = _resolve_model(model, max_tokens)
        self.tools: dict[str, Tool] = {}
        for obj in tools or []:
            tool = _as_tool(obj)
            self.tools[tool.name] = tool
        self.instructions = instructions
        self.max_steps = max_steps

    def _schemas(self) -> list[dict]:
        return [tool.to_schema() for tool in self.tools.values()]

    def _dispatch(self, call: ToolCall) -> str:
        tool = self.tools.get(call.name)
        if tool is None:
            return f"Error: unknown tool '{call.name}'"
        try:
            out = tool.func(**call.arguments)
        except Exception as exc:  # surface errors to the model, do not crash
            return f"Error: {exc}"
        return out if isinstance(out, str) else json.dumps(out, default=str)

    def stream(self, prompt: str, *, trace: bool = False,
               messages: list | None = None) -> t.Iterator[Event]:
        """Run the agent, yielding an Event per step. Returns a Result.

        Capture the return value with `run`, or iterate for live output:
            for event in agent.stream("hi"):
                ...
        """
        history: list = list(messages) if messages else []
        history.append({"role": "user", "content": prompt})

        for step in range(self.max_steps):
            resp = self.model.generate(
                system=self.instructions,
                messages=history,
                tools=self._schemas(),
            )
            history.append(resp.assistant_message)

            if trace:
                _print_model(resp.text)
            yield Event(type="model", text=resp.text)

            if not resp.tool_calls:
                if trace:
                    print()
                yield Event(type="final", text=resp.text)
                return Result(output=resp.text, messages=history, steps=step + 1)

            results: list[ToolResult] = []
            for call in resp.tool_calls:
                if trace:
                    _print_tool_call(call)
                yield Event(type="tool_call", name=call.name,
                            arguments=call.arguments)

                out = self._dispatch(call)

                if trace:
                    _print_tool_result(out)
                yield Event(type="tool_result", name=call.name, result=out)
                results.append(ToolResult(id=call.id, content=out))

            history.append(self.model.tool_result_message(results))

        text = f"(stopped: reached max_steps={self.max_steps})"
        yield Event(type="final", text=text)
        return Result(output=text, messages=history, steps=self.max_steps)

    def run(self, prompt: str, *, trace: bool = False,
            messages: list | None = None) -> Result:
        """Run to completion and return the Result."""
        gen = self.stream(prompt, trace=trace, messages=messages)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            return stop.value
