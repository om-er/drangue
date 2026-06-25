"""The whole pitch, runnable. Needs ANTHROPIC_API_KEY in your environment."""

from drangue import Agent, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 72F and sunny"


agent = Agent(
    model="claude-opus-4-8",
    tools=[get_weather],
    instructions="You are a concise weather assistant.",
)

if __name__ == "__main__":
    result = agent.run("What should I wear in Paris today?", trace=True)
    print("FINAL:", result.output)
