---
hide:
  - navigation
  - toc
title: LazyBridge — build complex pipelines with Claude Code and Codex
description: Interchangeable engines, independent review and deterministic project records. See real work built with LazyBridge.
---

<section class="lb-hero lb-manifesto">
  <div class="lb-hero__copy">
    <div class="lb-pill">Agent = Engine + Tools + State</div>
    <h1>Build complex pipelines with <span class="accent">Claude Code and Codex</span>.</h1>
    <p class="lb-subhead">LazyBridge makes coding runtimes engines of your Python agent. LazyTools connects their work into durable projects.</p>
    <ul class="lb-principles">
      <li>Swap subscription and API engines; keep tools and state.</li>
      <li>One writes, the other reviews. The producer never signs its own review.</li>
      <li>Code computes. The LLM reads the results.</li>
      <li>Use deterministic gates before anything ships.</li>
    </ul>
    <div class="lb-cta-row">
      <a href="#built-with" class="lb-btn lb-btn--primary">See what it builds &rarr;</a>
      <a href="#engines" class="lb-btn lb-btn--ghost">Read the code &rarr;</a>
    </div>
  </div>
  <figure class="lb-hero__proof">
    <a href="#built-with"><img src="assets/showcase/stazione-fronte.webp" alt="Victorian station built brick by brick by the LazyBricks pipeline" width="1600" height="1600" fetchpriority="high" /></a>
    <figcaption>LazyBricks · from brief and concept to an LDraw model</figcaption>
  </figure>
</section>

<div class="lb-demo lb-home-section" markdown="1">

## Swap the engine. Keep the agent. {#engines}

<div class="lb-demo__grid" markdown="1">
<div markdown="1">

**Claude Code, Codex and provider APIs share one Agent contract.**

With a subscription login, the coding engines use your plan’s quota. Your tools, memory and session stay in Python when you change engines.

[How it works →](https://core.lazybridge.com/guides/full/coding-agents/)



</div>
<div class="lb-demo__code" markdown="1">

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, LLMEngine, Memory, Session

def project_status() -> str:
    """Return this example project's status."""
    return "The draft is ready for review."

state = dict(tools=[project_status], memory=Memory(), session=Session())
agent = Agent(engine=ClaudeCodeEngine(), **state)
agent = Agent(engine=CodexEngine(), **state)  # same tools, memory and session
agent = Agent(engine=LLMEngine("gpt-6.1-sol"), **state)  # provider API
```

</div>
</div>

</div>

<div class="lb-demo lb-home-section" markdown="1">

## One writes. The other reviews. {#collaboration}

<div class="lb-demo__grid" markdown="1">
<div markdown="1">

**Make review a separate engine’s job.**

Give Claude Code the draft and Codex the review, or reverse the roles. The Agent verification loop brings reviewer feedback into the next attempt.

[How it works →](blog/posts/claude-writes-codex-reviews.md)



</div>
<div class="lb-demo__code" markdown="1">

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine

reviewer = Agent(name="codex-reviewer", engine=CodexEngine())
writer = Agent(name="claude-writer", engine=ClaudeCodeEngine(),
               verify=reviewer, max_verify=3)
result = writer("Draft a function that blocks weekend deployment.")
print(result.text())
```

</div>
</div>

</div>

<div class="lb-demo lb-home-section" markdown="1">

## Delegate work that takes time. {#async-bridge}

<div class="lb-demo__grid" markdown="1">
<div markdown="1">

**Detached jobs keep approvals and results in a durable queue.**

The code bridge supports run, pending, approve, result and cancel. Native sessions and threads resume by name; failed jobs retain workspace evidence.

[How it works →](https://tools.lazybridge.com/code-bridge/)



</div>
<div class="lb-demo__code" markdown="1">

```sh
lazytools-code-bridge run --detach --engine claude --cwd . --task "Implement the retry fix" --session retry-fix
lazytools-code-bridge pending --json
lazytools-code-bridge approve APPROVAL_ID
lazytools-code-bridge result WRITER_JOB_ID
lazytools-code-bridge run --detach --tier writing --review-of WRITER_JOB_ID --cwd . --task "Review the fix without editing"
lazytools-code-bridge cancel ACTIVE_JOB_ID --reason "Stop this attempt"
```

</div>
</div>

</div>

<div class="lb-demo lb-home-section" markdown="1">

## One project, shared state. {#projects}

<div class="lb-demo__grid" markdown="1">
<div markdown="1">

**Plans, claims and acceptance evidence live together.**

LazyTools adds contracts, verification records and a quota brake to the shared Store. Code renders the schedule as a deterministic Gantt.

[How it works →](https://tools.lazybridge.com/projects/)



</div>
<div class="lb-demo__code" markdown="1">

```python
from lazybridge import Store
from lazybridge.ext.planners import DurableBlackboard
from lazytools.projects import brake, owner, records

store = Store()  # use Store(db="projects.sqlite") to persist
records.open_project(store, project_id="retry-fix", title="Retry fix",
                     objective="Handle transient failures and verify boundary cases")
owner.set_project_owner(store, "retry-fix", "shared")
board = DurableBlackboard(store, plan_id="project:retry-fix")
board.set_plan("Write, then review with the opposite engine",
               ["Implement retry fix", "Review and run boundary checks"])
print(brake.get_project_brake_enabled(store, "retry-fix"))  # True
```

</div>
</div>
<figure class="lb-band-proof"><a href="assets/showcase/ceo-gantt.png"><img src="assets/showcase/ceo-gantt-detail.webp" alt="Docs project schedule in the LazyCEO Gantt overview" loading="lazy" /></a><figcaption>Gantt overview excerpt — rendered by the real GUI, demo data.</figcaption></figure>
</div>

<div class="lb-demo lb-home-section" markdown="1">

## Route by remaining quota. {#quota-routing}

<div class="lb-demo__grid" markdown="1">
<div markdown="1">

**Deterministic tiers choose an eligible engine and model.**

The router reads quota, applies the tier ladder and explains its choice. Review routing excludes the producing engine.

[How it works →](https://tools.lazybridge.com/code-bridge/#choose-a-model-from-quota)

<p class="lb-benchmark">Benchmark in progress</p>

</div>
<div class="lb-demo__code" markdown="1">

```sh
lazytools-code-bridge route --tier writing --cwd . --json
```

</div>
</div>
<figure class="lb-band-proof"><a href="assets/showcase/ceo-brake.png"><img src="assets/showcase/ceo-brake-history.webp" alt="LazyCEO Brake page excerpt showing separate Codex and Claude Code quota-history charts" loading="lazy" /></a><figcaption>Per-engine quota history excerpt — rendered by the real GUI, demo data.</figcaption></figure>
</div>

<div class="lb-demo lb-home-section" markdown="1">

## LazyTools connects the work. {#lazytools}

<div class="lb-demo__grid" markdown="1">
<div markdown="1">

**Expose the same project and coding tools through MCP.**

The integration layer brings together the code bridge, projects, routing and connectors. Tools include codex_code_review, codex_write and claude_*.

[How it works →](https://tools.lazybridge.com/mcp-server/)

<p class="lb-version">LazyBridge 1.8.0 · core Stable &nbsp; / &nbsp; LazyTools 0.10.1 · Alpha</p>

</div>
<div class="lb-demo__code" markdown="1">

```sh
lazytools-mcp projects code_review claude_review code_write code_sessions --allow-unsafe
```

</div>
</div>

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
<p>A CEO agent that runs projects across the workspace and delegates to Codex and Claude Code.<br />
Durable project records feed a deterministic Gantt and individual Program pages.<br />
A quota brake for each engine governs autonomous task admission; its GUI shows quota history.</p>
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
