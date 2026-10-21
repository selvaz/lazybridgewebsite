---
hide:
  - navigation
  - toc
title: "LazyBridge — one agent, any engine: subscription or API"
description: "Build agents and pipelines in Python with any model, and command your whole fleet from one chat — on your Claude Code or Codex subscription, or any API model per token."
---

<section class="lb-hero lb-manifesto lb-hero--banner">
  <div class="lb-hero__art"><img src="assets/hero.webp" alt="LazyBridge mascot: a kid running from an idea, through AI, tools and data, to a rocket launch" width="1672" height="941" fetchpriority="high" /></div>
  <div class="lb-hero__copy">
    <div class="lb-pill">Open-source Python framework · Apache 2.0</div>
    <h1>Build AI agents and pipelines in Python, <span class="accent">with any model</span>.</h1>
    <p class="lb-subhead">LazyBridge is an orchestration framework: you compose agents, tools and deterministic steps into pipelines that run on Claude Code or Codex with your subscription, or on any API model billed per token.</p>
    <p class="lb-motto">One framework. Command your fleet yourself, or hand it to an LLM assistant.</p>
    <p class="lb-apart-title">What sets it apart</p>
    <ul class="lb-principles">
      <li><strong>Everything is a tool.</strong> No graph to wire, no crew of roles: functions, agents and whole pipelines plug into each other, so pipelines nest inside pipelines.</li>
      <li><strong>Your fleet, under control.</strong> Direct work across projects from one chat, or delegate it to an organizer agent; each task goes to the engine that fits it, with quota and costs kept in check.</li>
      <li><strong>Deterministic, agentic or both.</strong> Fix the order with a Plan where the process must be repeatable, let agents decide where judgment helps, and mix the two in one pipeline.</li>
      <li><strong>Subscription or API, same Agent.</strong> Swap Claude Code, Codex or an API model in one line; tools and state stay.</li>
    </ul>
    <div class="lb-cta-row">
      <a href="#why" class="lb-btn lb-btn--primary">See how it works &rarr;</a>
      <a href="#built-with" class="lb-btn lb-btn--ghost">See what it builds &rarr;</a>
    </div>
    <p class="lb-install"><code>pip install lazybridge</code></p>
  </div>

</section>

<div class="lb-home-section lb-compare" markdown="1">

## Not a graph. Not a crew. Agents that nest. {#compare}

Graph frameworks such as LangGraph model a workflow as nodes and edges over a shared state, compiled into a graph. Crew frameworks such as CrewAI assign roles and tasks to a crew of agents. **LazyBridge has one building block: the agent** (engine + tools + state). A pipeline is an agent too, with a deterministic **Plan** as its engine when the order must be fixed, so you compose by putting agents inside agents.

| | You model a workflow as | Main concepts |
|---|---|---|
| **LangGraph** | a state graph of nodes and edges, compiled | graph, state, node, edge |
| **CrewAI** | a crew of agents with roles, running tasks in a process | agent, role, task, crew, process |
| **LazyBridge** | agents inside agents; a Plan when the order must be fixed | agent, engine, tool, plan |

<p class="lb-compare__note">Summary of each project’s documented programming model. <a href="https://core.lazybridge.com/best-python-ai-agent-frameworks/">Full comparison with sources →</a></p>

</div>

<div class="lb-demo lb-home-section lb-why" markdown="1">

## One building block. Endless combinations. {#why}

<div class="lb-demo__grid" markdown="1">
<div markdown="1">

<span id="collaboration"></span>**LazyBridge began as a bridge between LLMs from different providers.** A model rarely spots its own mistakes; a model trained differently often does. So in LazyBridge they work together and review each other’s work, catching hallucinations and errors before they reach the result.

**One building block: Agent = Engine + Tools + State.** Everything is a tool: a function, another agent, a whole pipeline. The engine can be anything: an API model, Claude Code or Codex on your subscription, a human, or a deterministic **Plan**. Every level keeps the same contract (tools, memory, verification, one Envelope out), so pipelines nest inside pipelines.

**Build a team once. Reuse it as one tool in the next thing you build.**

[Why LazyBridge →](https://core.lazybridge.com/why/)

</div>
<div class="lb-demo__code" markdown="1">

```python
from lazybridge import (Agent, ClaudeCodeEngine, CodexEngine,
                        LLMEngine, Plan, Step)

# OpenAI researches (API); Claude Code writes (subscription);
# Codex, another provider, reviews the draft before it moves on.
research = Agent(name="research", engine=LLMEngine("gpt-6.1-sol"))
write = Agent(name="write", engine=ClaudeCodeEngine(),
              verify=Agent(name="review", engine=CodexEngine()))

# A deterministic Plan is just another engine; agents are its tools.
report = Agent(name="report", engine=Plan(Step("research"), Step("write")),
               tools=[research, write])

# The whole pipeline is now one tool for the next agent.
desk = Agent(engine=LLMEngine("claude-opus-5-5"), tools=[report])
```

<div class="lb-nest" aria-label="Diagram: an Agent whose tools include a Plan-driven Agent, whose tools are agents from different providers with a cross-provider reviewer">
<div class="lb-nest__box"><span class="lb-nest__tag">Agent · desk</span><span class="lb-nest__engine">engine: Claude Opus (API)</span>
<div class="lb-nest__box"><span class="lb-nest__tag">Agent · report</span><span class="lb-nest__engine">engine: Plan (deterministic)</span>
<div class="lb-nest__row">
<div class="lb-nest__leaf">research<br><small>OpenAI · API</small></div>
<div class="lb-nest__leaf">write<br><small>Claude Code · subscription</small><br><small class="lb-nest__review">⟲ reviewed by Codex</small></div>
</div>
</div>
</div>
</div>

</div>
</div>

</div>

<div class="lb-demo lb-home-section" markdown="1">

## Subscription or tokens: same Agent. {#engines}

<div class="lb-feature-cards lb-engine-parity">
  <div class="lb-card"><div class="lb-card__body"><h3>Claude Code</h3><p><strong>Subscription</strong> · your Claude plan quota.</p><p><code>ClaudeCodeEngine()</code></p></div></div>
  <div class="lb-card"><div class="lb-card__body"><h3>Codex</h3><p><strong>Subscription</strong> · your ChatGPT plan quota.</p><p><code>CodexEngine()</code></p></div></div>
  <div class="lb-card"><div class="lb-card__body"><h3>API models</h3><p><strong>Per token</strong> · Anthropic, OpenAI, Google, DeepSeek, LiteLLM.</p><p><code>LLMEngine("model-id")</code></p></div></div>
</div>

<div class="lb-demo__grid" markdown="1">
<div markdown="1">

**One Agent contract for every engine.** Tools, memory, session and verification are LazyBridge objects, not engine features: change the engine and the rest of your pipeline stays the same. Mix them freely — a subscription agent can write while an API model reviews, or the other way round.

The login decides billing: a subscription login uses the plan's quota, an API key is billed per token.

[How it works →](https://core.lazybridge.com/guides/full/coding-agents/)

</div>
<div class="lb-demo__code" markdown="1">

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, LLMEngine, Memory, Session

def project_status() -> str:
    """Return this example project's status."""
    return "The draft is ready for review."

state = dict(tools=[project_status], memory=Memory(), session=Session())

# Subscription agents: your plan's quota
claude = Agent(engine=ClaudeCodeEngine(), **state)
codex = Agent(engine=CodexEngine(), **state)

# API models: billed per token
opus = Agent(engine=LLMEngine("claude-opus-5-5"), **state)
gpt = Agent(engine=LLMEngine("gpt-6.1-sol"), **state)
```

</div>
</div>

</div>

<div class="lb-demo lb-home-section" markdown="1">

## No Python required: run the fleet from Claude Code or Codex. {#mcp}

<div class="lb-demo__grid" markdown="1">
<div markdown="1">

**Plug it into the tools your team already uses.** Add the LazyTools MCP server to Claude Code (CLI, desktop or IDE) or to Codex, and your assistant gains tools to work with the other one: Claude can ask Codex to write or review code, Codex can ask Claude, and both share project plans, a deterministic Gantt and a quota brake.

Specialist agents built with LazyBridge can run on other API models too, so the fleet of agents you direct from one chat is not limited to one provider.

[MCP server setup →](https://tools.lazybridge.com/mcp-server/)

</div>
<div class="lb-demo__code" markdown="1">

```sh
# Claude Code: Codex becomes a tool
claude mcp add lazytools -- lazytools-mcp projects code_review code_write \
    code_sessions --allow-unsafe --config lazytools.json
```

```toml
# Codex (~/.codex/config.toml): Claude becomes a tool
[mcp_servers.lazytools]
command = "lazytools-mcp"
args = ["projects", "claude_review", "code_write", "code_sessions",
        "--allow-unsafe", "--config", "lazytools.json"]
```

</div>
</div>

</div>

<div class="lb-home-section" markdown="1">

## From one agent to a fleet. {#ecosystem}

<div class="lb-feature-cards lb-ecosystem-cards">
  <div class="lb-card"><div class="lb-card__body"><h3>Async code bridge</h3><p>Run Claude Code and Codex jobs in the background, approve their requests, route the review to the other engine.</p><a href="https://tools.lazybridge.com/code-bridge/" class="lb-card__link">How it works &rarr;</a></div></div>
  <div class="lb-card"><div class="lb-card__body"><h3>Fleets across projects</h3><p>Run fleets of agents on many projects at once: plans, claims and acceptance evidence in one Store, a deterministic Gantt for every project.</p><a href="https://tools.lazybridge.com/projects/" class="lb-card__link">How it works &rarr;</a></div></div>
  <div class="lb-card"><div class="lb-card__body"><h3>Quota routing</h3><p>Spread the fleet’s work by each engine’s remaining subscription quota; a brake stops work past the limits.</p><a href="https://tools.lazybridge.com/code-bridge/#choose-a-model-from-quota" class="lb-card__link">How it works &rarr;</a></div></div>
  <div class="lb-card"><div class="lb-card__body"><h3>Durable sessions</h3><p>Name native Claude Code and Codex conversations and resume them across processes and restarts.</p><a href="https://tools.lazybridge.com/code-support/" class="lb-card__link">How it works &rarr;</a></div></div>
</div>

<p class="lb-benchmark">Benchmark in progress: cross-provider review vs a single model, and quota routing.</p>

<p class="lb-version">LazyBridge 1.8.0 · core Stable &nbsp; / &nbsp; LazyTools 0.10.1 · Alpha</p>

</div>

<div class="lb-home-section lb-showcase" markdown="1">

## Built with LazyBridge {#built-with}

Working pipelines, real outputs. Three private projects put the engines and shared tools to work.

<article class="lb-project" id="lazybricks">
<div class="lb-project__heading"><h3>LazyBricks</h3><span class="lb-private">Private repository</span></div>
<p>A local 3D brick-model editor built with Three.js, FastAPI and SQLite, indexed over the full LDraw Official Parts Library.<br />
The station below starts from a brief and a concept, then builds brick by brick: volumetric → simple → detailed, with quality reviews at each phase.<br />
The phone interface and MCP server let a person and an agent work on the same projects and revisions.</p>
<div class="lb-gallery lb-gallery--bricks">
<figure class="lb-gallery__lead"><a href="assets/showcase/stazione-fronte-4K.png"><img src="assets/showcase/stazione-fronte.webp" alt="Front view of the Victorian station with clock tower and brick facade" loading="lazy" /></a><figcaption>Victorian station · front view · open the 4K original</figcaption></figure><figure class=""><a href="assets/showcase/stazione-esplosa-4K.png"><img src="assets/showcase/stazione-esplosa.webp" alt="Exploded storeys of the station showing the tower, roof and two floors" loading="lazy" /></a><figcaption>Exploded storeys · 4K original</figcaption></figure><figure class=""><a href="assets/showcase/stazione-interni-4K.png"><img src="assets/showcase/stazione-interni.webp" alt="Station ground-floor interior with counter, seating and platform" loading="lazy" /></a><figcaption>Interiors · 4K original</figcaption></figure><figure class=""><a href="assets/showcase/stazione-binari-4K.png"><img src="assets/showcase/stazione-binari.webp" alt="Victorian station viewed from the tracks and covered platform" loading="lazy" /></a><figcaption>Tracks and platform · 4K original</figcaption></figure></div><p class="lb-project-contact">Interested in collaborating? Write to <a href="mailto:lazybridgelabs@gmail.com">lazybridgelabs@gmail.com</a>.</p></article>
<article class="lb-project" id="lazycommittee">
<div class="lb-project__heading"><h3>LazyCommittee</h3><span class="lb-private">Private repository</span></div>
<p>An AI investment committee built on LazyBridge with Claude Code and Codex as engines.<br />
Scheduled daily digests become a weekly world state, asset-class views and a portfolio document.<br />
Code computes the numbers; the LLM reads results. Freshness, numeric lint and source-map checks gate publication.</p>
<div class="lb-gallery lb-gallery--reports">
<figure class=""><a href="assets/showcase/committee-digest-eco.png"><img src="assets/showcase/committee-digest-eco.webp" alt="Real economics digest excerpt with dated values, percentiles, source references and a source list" loading="lazy" /></a><figcaption>Daily economics digest excerpt · each number is dated and sourced, with its percentile.</figcaption></figure><figure class=""><a href="assets/showcase/committee-portfolio-tree.png"><img src="assets/showcase/committee-portfolio-tree.webp" alt="Hierarchical asset-class allocation tree from the portfolio optimizer report" loading="lazy" /></a><figcaption>Portfolio optimizer · hierarchical allocation tree: asset classes, regions and the ETFs under each node.</figcaption></figure></div><p class="lb-project-contact">Interested in collaborating? Write to <a href="mailto:lazybridgelabs@gmail.com">lazybridgelabs@gmail.com</a>.</p></article>
<article class="lb-project" id="lazyceo">
<div class="lb-project__heading"><h3>LazyCEO</h3><span class="lb-private">Private repository</span></div>
<p>A CEO agent that runs a fleet of agents across many projects at once, delegating to Codex and Claude Code.<br />
Every project has a deterministic Gantt; the portfolio of running projects is tracked side by side.<br />
A quota brake per engine routes and paces the fleet’s work; its GUI shows quota history.</p>
<div class="lb-gallery lb-gallery--ceo">
<figure class="lb-gallery__lead"><a href="assets/showcase/ceo-brake-history.png"><img src="assets/showcase/ceo-brake-history.webp" alt="LazyCEO quota brake: Codex and Claude Code usage over the last 14 days, with the autonomous and ceiling thresholds and a weekly reset" loading="lazy" /></a><figcaption>Quota brake per engine, with 14 days of history — rendered by the real GUI, demo data.</figcaption></figure><figure class="lb-gallery__lead"><a href="assets/showcase/ceo-gantt-detail.png"><img src="assets/showcase/ceo-gantt-detail.webp" alt="LazyCEO program Gantt: completed tasks, tasks in progress and planned tasks around today" loading="lazy" /></a><figcaption>Project Gantt: done, in progress and planned — rendered by the real GUI, demo data.</figcaption></figure></div><p class="lb-project-contact">Interested in collaborating? Write to <a href="mailto:lazybridgelabs@gmail.com">lazybridgelabs@gmail.com</a>.</p></article>

[More about these projects →](built-with.md)

</div>
<div class="lb-home-cta" markdown="1">

## Build your next pipeline.

Interested in collaborating? Write to [lazybridgelabs@gmail.com](mailto:lazybridgelabs@gmail.com).

Install into your project environment, then sign in to your local coding runtimes:

```sh
python -m pip install "lazybridge[claude-code,openai]==1.8.0"
python -m pip install "lazytoolkit[engines,mcp] @ git+https://github.com/selvaz/LazyTools.git@v0.10.1"
```

[Engine setup →](https://core.lazybridge.com/guides/full/coding-agents/) · [MCP and workspace configuration →](https://tools.lazybridge.com/mcp-server/)

</div>
