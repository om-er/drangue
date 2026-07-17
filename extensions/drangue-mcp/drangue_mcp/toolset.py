"""MCP tools for drangue: any Model Context Protocol server's tools, wrapped.

An MCP server advertises tools with JSON schemas; drangue tools ARE JSON
schemas plus a callable. The adapter is therefore a passthrough: list the
server's tools, wrap each as a `drangue.Tool` whose func proxies the call
back over the session, and hand the list to an Agent. Guardrails, autonomy,
hardening policies, and the event log all apply to MCP tools exactly as they
do to local ones, because by the time the runtime sees them they are ordinary
tools.

    from drangue import Agent
    from drangue_mcp import stdio_toolset

    async with stdio_toolset("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"]) as toolset:
        agent = Agent("claude-opus-4-8", tools=toolset.tools)
        result = await agent.run("what is in /data?")

Anything session-shaped works: `tools_from_session` needs only async
`list_tools()` and `call_tool(name, arguments)`, so tests run against a fake
with no server and no network.
"""

from __future__ import annotations

import json
import typing as t
from contextlib import asynccontextmanager

from drangue.errors import ToolError
from drangue.tool import Tool


def _render_content(blocks) -> str:
    """Flatten MCP content blocks to the text a model consumes."""
    parts: list[str] = []
    for b in blocks or []:
        kind = getattr(b, "type", None)
        if kind == "text":
            parts.append(getattr(b, "text", ""))
        else:
            # Honest placeholder: dropping it silently would misrepresent the
            # tool's answer; inventing a rendering would misrepresent it more.
            parts.append(json.dumps({"type": kind or "unknown",
                                     "note": "non-text content omitted"}))
    return "\n".join(p for p in parts if p)


def _wrap(session, spec) -> Tool:
    name = getattr(spec, "name", None) or spec["name"]
    description = getattr(spec, "description", None) or ""
    schema = getattr(spec, "inputSchema", None) or {"type": "object",
                                                    "properties": {}}

    async def call(**kwargs):
        result = await session.call_tool(name, arguments=kwargs)
        text = _render_content(getattr(result, "content", None))
        if getattr(result, "isError", False):
            # Surfaces through drangue's hardening as a clean, categorized
            # failure the model can reason about, like any local tool error.
            raise ToolError(text or f"MCP tool {name!r} failed")
        return text

    return Tool(name=name, description=description, parameters=schema, func=call)


async def tools_from_session(session) -> list[Tool]:
    """Wrap every tool an MCP session advertises as a drangue Tool."""
    listed = await session.list_tools()
    return [_wrap(session, spec) for spec in listed.tools]


class MCPToolset:
    """A connected MCP server's tools, ready to hand to an Agent."""

    def __init__(self, session, tools: list[Tool]):
        self.session = session
        self.tools = tools

    @classmethod
    async def from_session(cls, session) -> "MCPToolset":
        return cls(session, await tools_from_session(session))


@asynccontextmanager
async def stdio_toolset(command: str, args: t.Sequence[str] = (),
                        env: dict | None = None):
    """Connect to a stdio MCP server for the duration of the block.

    Imports the `mcp` SDK lazily so the package imports cleanly without it;
    only actually connecting requires it.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise ImportError(
            "The MCP SDK is required to connect to a server. "
            "Install it with: pip install drangue-mcp"
        ) from exc

    params = StdioServerParameters(command=command, args=list(args), env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield await MCPToolset.from_session(session)
