# drangue-mcp

Model Context Protocol servers as drangue tool packs (a Tier 3 battery on the
Tools seam, see `ECOSYSTEM.md`).

An MCP server advertises tools with JSON schemas; drangue tools are JSON
schemas plus a callable. This adapter lists a server's tools and wraps each as
an ordinary `drangue.Tool`, so guardrails, autonomy modes, hardening policies,
and the event log apply to them unchanged.

```python
from drangue import Agent
from drangue_mcp import stdio_toolset

async with stdio_toolset("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"]) as toolset:
    agent = Agent("claude-opus-4-8", tools=toolset.tools)
    result = await agent.run("what is in /data?")
```

Already holding a connected `mcp.ClientSession` (or anything with async
`list_tools()` and `call_tool(name, arguments)`)? Wrap it directly:

```python
from drangue_mcp import tools_from_session

tools = await tools_from_session(session)
```

Behavior notes:

- A server-side tool error (`isError`) surfaces through drangue's hardening as
  a clean, categorized failure the model can reason about, like any local
  tool failure.
- Non-text content blocks (images, embedded resources) are flagged with an
  honest placeholder rather than silently dropped; text blocks pass through.
- The package imports without the `mcp` SDK; only connecting requires it.

Install: `pip install drangue-mcp`. Tests run fully offline against a fake
session: `python extensions/run_tests.py`.
