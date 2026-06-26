"""Phase 4 tests: budgets, routing, model recording, and prompt caching.

All offline with scripted models and a fake Anthropic client.
"""

from types import SimpleNamespace

from drangue import (
    Agent,
    AnthropicModel,
    Budget,
    ModelResponse,
    RuleRouter,
    ToolCall,
    tool,
)
from drangue.models import Model


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class LoopingModel(Model):
    """Always asks for a tool, reporting fixed usage; loops until stopped."""

    def __init__(self, name="m", per_call_tokens=100):
        self.model = name
        self.per_call_tokens = per_call_tokens
        self.calls = 0

    async def generate(self, *, system, messages, tools):
        self.calls += 1
        return ModelResponse(
            tool_calls=[ToolCall(f"c{self.calls}", "add", {"a": 1, "b": 1})],
            usage={"input_tokens": self.per_call_tokens, "output_tokens": 0},
        )


async def test_token_budget_stops_before_an_unaffordable_step():
    model = LoopingModel(per_call_tokens=100)
    agent = Agent(model=model, tools=[add], budget=Budget(max_tokens=150))
    result = await agent.run("go")

    # After call 1 spend is 100 (< 150, allowed). After call 2 spend is 200,
    # so the third model call is refused before it starts.
    assert model.calls == 2
    assert "budget" in result.output


async def test_usd_budget_uses_prices_and_recorded_model_name():
    model = LoopingModel(name="pricey", per_call_tokens=1_000_000)
    prices = {"pricey": {"input": 10.0, "output": 30.0}}   # $10 per 1M input tokens
    agent = Agent(model=model, tools=[add],
                  budget=Budget(max_usd=15.0, prices=prices))
    result = await agent.run("go")

    # Each call costs $10 of input. After 2 calls ($20) the next is refused.
    assert model.calls == 2
    assert "budget" in result.output


async def test_model_name_is_recorded_in_the_log():
    model = LoopingModel(name="claude-test")
    agent = Agent(model=model, tools=[add], max_steps=1)
    result = await agent.run("go")
    decisions = [e for e in result.events if e.type == "model_decision"]
    assert decisions[0].payload["model"] == "claude-test"


async def test_rule_router_picks_model_per_step():
    class Named(Model):
        def __init__(self, name, steps):
            self.model = name
            self.steps = steps
            self.i = 0
            self.calls = 0

        async def generate(self, *, system, messages, tools):
            self.calls += 1
            resp = self.steps[self.i]
            self.i += 1
            return resp

    smart = Named("smart", [ModelResponse(tool_calls=[ToolCall("c1", "add", {"a": 1, "b": 1})])])
    cheap = Named("cheap", [ModelResponse(text="done")])

    # First model call (index 0) -> smart; everything after -> cheap.
    router = RuleRouter(default=cheap, rules=[(lambda msgs, i: i == 0, smart)])
    result = await Agent(model=router, tools=[add]).run("go")

    assert result.output == "done"
    assert smart.calls == 1
    assert cheap.calls == 1
    models_used = [e.payload["model"] for e in result.events if e.type == "model_decision"]
    assert models_used == ["smart", "cheap"]


class FakeAnthropic:
    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[], usage=None, stop_reason="end_turn")


async def test_cache_marks_the_stable_prefix():
    fake = FakeAnthropic()
    model = AnthropicModel("claude-test", client=fake, cache=True)
    await model.generate(system="rules", messages=[{"role": "user", "content": "hi"}],
                         tools=[add])

    sent = fake.calls[0]
    assert isinstance(sent["system"], list)
    assert sent["system"][0]["cache_control"]["type"] == "ephemeral"
    assert sent["tools"][-1]["cache_control"]["type"] == "ephemeral"


async def test_no_cache_by_default():
    fake = FakeAnthropic()
    model = AnthropicModel("claude-test", client=fake)   # cache defaults to off
    await model.generate(system="rules", messages=[{"role": "user", "content": "hi"}],
                         tools=[add])

    sent = fake.calls[0]
    assert sent["system"] == "rules"
    assert "cache_control" not in sent["tools"][-1]
