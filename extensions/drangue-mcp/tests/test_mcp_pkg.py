"""Offline tests for drangue-mcp: a fake session, no server, no network."""

import json
from types import SimpleNamespace

from drangue import Agent, ModelResponse, ToolCall
from drangue.models import Model
from drangue_mcp import MCPToolset, tools_from_session


def _spec(name, description="", schema=None):
    return SimpleNamespace(name=name, description=description,
                           inputSchema=schema or {"type": "object",
                                                  "properties": {}})


def _text(text):
    return SimpleNamespace(type="text", text=text)


class FakeSession:
    def __init__(self, specs, responses):
        self.specs = specs
        self.responses = responses      # name -> result
        self.calls = []

    async def list_tools(self):
        return SimpleNamespace(tools=self.specs)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.responses[name]


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = steps
        self.i = 0
        self.seen_tools = []

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        self.seen_tools.append({t.name: t.to_schema() for t in tools})
        resp = self.steps[self.i]
        self.i += 1
        return resp


async def test_server_tools_become_drangue_tools_with_passthrough_schemas():
    schema = {"type": "object", "properties": {"path": {"type": "string"}},
              "required": ["path"]}
    session = FakeSession(
        [_spec("read_file", "Read a file.", schema)],
        {"read_file": SimpleNamespace(content=[_text("contents")], isError=False)},
    )
    tools = await tools_from_session(session)

    assert [t.name for t in tools] == ["read_file"]
    assert tools[0].to_schema() == {
        "name": "read_file",
        "description": "Read a file.",
        "input_schema": schema,
    }


async def test_an_agent_drives_mcp_tools_like_local_ones():
    session = FakeSession(
        [_spec("lookup")],
        {"lookup": SimpleNamespace(content=[_text("42")], isError=False)},
    )
    toolset = await MCPToolset.from_session(session)

    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "lookup", {"q": "answer"})]),
        ModelResponse(text="it is 42"),
    ])
    result = await Agent(model=model, tools=toolset.tools).run("go")

    assert result.output == "it is 42"
    assert session.calls == [("lookup", {"q": "answer"})]
    tr = [e for e in result.events if e.type == "tool_result"][0]
    assert tr.payload["content"] == "42"
    # The model saw the server's schema.
    assert "lookup" in model.seen_tools[0]


async def test_server_errors_surface_as_clean_categorized_failures():
    session = FakeSession(
        [_spec("flaky")],
        {"flaky": SimpleNamespace(content=[_text("backend exploded")], isError=True)},
    )
    toolset = await MCPToolset.from_session(session)

    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "flaky", {})]),
        ModelResponse(text="handled"),
    ])
    result = await Agent(model=model, tools=toolset.tools).run("go")

    assert result.output == "handled"
    tr = [e for e in result.events if e.type == "tool_result"][0]
    parsed = json.loads(tr.payload["content"])
    assert parsed["ok"] is False
    assert "backend exploded" in parsed["error"]["message"]


async def test_non_text_content_is_flagged_not_dropped():
    session = FakeSession(
        [_spec("shot")],
        {"shot": SimpleNamespace(
            content=[_text("caption"), SimpleNamespace(type="image", data="...")],
            isError=False)},
    )
    tools = await tools_from_session(session)
    out = await tools[0].func()
    assert "caption" in out
    assert "non-text content omitted" in out
