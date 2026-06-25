"""Run the same agent against a cheap (or free) OpenAI-compatible backend.

The cheapest option is local and free: install Ollama (https://ollama.com),
run `ollama pull llama3.1`, and no API key or per-token cost is involved.

To use a cheap hosted backend instead, swap the model line below, for example:
    model=OpenAIModel("deepseek-chat", base_url="https://api.deepseek.com")
    model=OpenAIModel("llama-3.1-8b-instant",
                      base_url="https://api.groq.com/openai/v1")
"""

from drangue import Agent, OpenAIModel, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 72F and sunny"


agent = Agent(
    # Free, local, no API key. Any value works as the key for Ollama.
    model=OpenAIModel(
        "llama3.1",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
    tools=[get_weather],
    instructions="You are a concise weather assistant.",
)

if __name__ == "__main__":
    result = agent.run("What should I wear in Paris today?", trace=True)
    print("FINAL:", result.output)
