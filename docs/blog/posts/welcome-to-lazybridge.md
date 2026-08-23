---
date: 2026-08-23
---

# Welcome to LazyBridge

!!! info "Written by LazyBridge agents, reviewed by a human before publishing"
    This post came out of a LazyBridge agent pipeline — research, draft, and
    editorial pass, each a separate agent — and was approved by a human before
    it went live. More on why in the [blog's About note](../index.md).

You know the moment. You've got a Python function — three lines, one docstring, done — and now you need to hand-write its JSON schema so an LLM can call it. `"type": "object"`, `"properties"`, `"required"`, the whole ceremony, duplicating information that's already sitting right there in your type hints. You copy the shape from the last tool you wrote, rename the fields, and immediately misspell `"parameters"` as `"parameter"` for the second time this month.

Then you get the schema working, and the *next* annoying thing shows up: the tool-calling loop. Call the model. Check `stop_reason`. If it's `tool_use`, pull out the blocks, run the right function for each one, thread the results back into the message list with the correct role, call the model again, repeat until it actually answers you. Do this for Anthropic's response shape, then again for OpenAI's slightly different one, then again when you try Gemini and none of it lines up.

None of this is hard. It's just boilerplate you've now written five times across five projects, and it's the same boilerplate whether you're calling a weather API or orchestrating a whole team of sub-agents.

That's the gap LazyBridge is for.

## What LazyBridge actually does

LazyBridge is a Python framework for building LLM agents — one `Agent` class, one tool contract, and providers you can swap without rewriting your integration code. A plain Python function, another `Agent`, or an MCP server can all go in the same `tools=[...]` list. LazyBridge turns them into something the model can call, and turns the model's response into something your code can call back.

Schematically, the whole surface looks like this:

```python
from lazybridge import Agent, LLMEngine

agent = Agent(engine=LLMEngine("claude-opus-5"))
result = agent("hello")
print(result.text())
```

That's genuinely the whole surface to start. No separate "graph" abstraction to learn before you can say hello. (The full, runnable version — with the API-key handling you actually need — is below.)

## The mental model: Engine + Tools + State

Every `Agent` is three things composed together, and once this clicks, the rest of the framework is just filling in the blanks.

- **Engine** — the thing that decides what happens next. Most of the time that's `LLMEngine`, wrapping a model provider. But it's swappable: a `Plan` engine follows a fixed sequence you define instead of letting the model improvise, a `HumanEngine` pauses for a person to approve a step, a `SupervisorEngine` gives you a REPL to drive things interactively.
- **Tools** — anything the agent is allowed to invoke. A plain function, another `Agent` (yes, agents can call agents), or a whole MCP server's worth of remote tools. They all live in the same list and get the same treatment.
- **State** — how the agent remembers things: `Memory` for in-prompt history, `Store` for a durable blackboard, `Session` as an event bus you can observe.

Here's why that matters in practice: the same `Agent(engine=..., tools=..., ...)` shape works for a quick one-off script and for something much more involved. You're mostly just changing what you pass to `engine=`, not learning a new API.

## A real example, first post edition

Here's the thing from earlier — a function becoming a tool, no manual schema — actually wired up to call a provider. It's written so it runs cleanly whether or not you have an API key sitting in your environment, because that's exactly the kind of code you'll want when you're first poking at something new:

```python
import os
from lazybridge import Agent, LLMEngine, Tool


def get_weather(city: str) -> str:
    """Return current temperature and conditions for `city`."""
    return f"{city}: 22°C, sunny"


agent = Agent(
    engine=LLMEngine("claude-opus-5"),
    tools=[Tool.wrap(get_weather, name="get_weather")],
)

if os.environ.get("ANTHROPIC_API_KEY"):
    result = agent("what's the weather in Rome and Paris?")
    print(result.text())
else:
    print("No ANTHROPIC_API_KEY set — skipping the live call.")
    print("Set one and rerun: get_weather becomes a callable tool with zero schema code.")
```

Run `pip install "lazybridge[anthropic]"`, drop that in a file, and try it both ways — with `ANTHROPIC_API_KEY` unset, and then with a real key exported. Same code, no crash either way. And when the key *is* there, LazyBridge reads the type hints and docstring off `get_weather` and builds the schema for you. That's one whole category of boilerplate from the top of this post, gone.

## What's next

This is the first post on this blog, so take it as an introduction rather than a tour — there's a lot of LazyBridge I haven't touched yet: giving an agent real memory across turns, wiring up MCP servers as tool catalogues, having agents call other agents and watching parallel tool calls run automatically, and building deterministic `Plan`-based pipelines when you want less improvisation and more guarantees. Future posts will go through those one at a time, with the same rule as today: working code, honest about what it does and doesn't do.