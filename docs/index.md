---
hide:
  - navigation
  - toc
title: LazyBridge — Python composition runtime for LLM workflows
description: Build LLM workflows where functions, agents, plans, humans and external systems all compose through one tool interface.
---

<!-- ═══════════════════ HERO ═══════════════════ -->
<section class="lb-hero">
  <div class="lb-hero__copy">
    <div class="lb-pill">Everything is a tool &bull; Python-first &bull; Composition runtime</div>

    <h1>LazyBridge is the Python framework for <span class="accent">composable LLM workflows</span>.</h1>

    <p class="lb-subhead">
      Functions, agents, deterministic plans, humans and external systems
      all compose through one tool interface &mdash; with validation,
      checkpointing and tracing built in.
    </p>

    <div class="lb-cta-row">
      <a href="https://core.lazybridge.com/quickstart/" class="lb-btn lb-btn--primary">Get started &rarr;</a>
      <a href="https://core.lazybridge.com/concepts/layered-composition/" class="lb-btn lb-btn--ghost">See the model</a>
    </div>

    <div class="lb-chips">
      <span class="lb-chip">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>
        Multi-provider
      </span>
      <span class="lb-chip">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
        Python-first
      </span>
      <span class="lb-chip">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
        Plan-based
      </span>
      <span class="lb-chip">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></svg>
        Observable
      </span>
      <span class="lb-chip">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
        Human-in-the-loop
      </span>
    </div>
  </div>

  <div class="lb-hero__art">
    <img src="assets/hero.png" alt="LazyBridge mascot" />
  </div>
</section>

<!-- ═══════════════════ FEATURE CARDS ═══════════════════ -->
<section class="lb-feature-cards">
  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>
    </div>
    <div class="lb-card__body">
      <h3>One abstraction, not five</h3>
      <p>Most agent frameworks split your system into chains, graphs, tools, agents, routers, and executors. LazyBridge collapses them into one recursive model: everything is a tool.</p>
      <a href="https://core.lazybridge.com/concepts/layered-composition/" class="lb-card__link">See how it composes &rarr;</a>
    </div>
  </div>

  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>
    </div>
    <div class="lb-card__body">
      <h3>Fail at build time, not at 3am</h3>
      <p>Deterministic plans catch duplicate steps, broken references, type drift, and invalid sentinels before the first LLM call runs.</p>
      <a href="https://core.lazybridge.com/guides/full/plan/" class="lb-card__link">See validation &rarr;</a>
    </div>
  </div>

  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></svg>
    </div>
    <div class="lb-card__body">
      <h3>Nested systems stay traceable</h3>
      <p>Agents can call plans, plans can call agents, and sub-agents can become tools &mdash; while sessions, event logs, costs, and OpenTelemetry spans roll up automatically.</p>
      <a href="https://core.lazybridge.com/guides/mid/session/" class="lb-card__link">See Session &rarr;</a>
    </div>
  </div>
</section>

<!-- ═══════════════════ DIAGRAM DECK (unified) ═══════════════════ -->
<div class="lb-concepts" markdown="1">

<div class="lb-concepts__header">
<h2>Under the hood</h2>
<p class="lb-concepts__sub">From a one-liner to governed multi-agent pipelines &mdash; one composition model throughout.</p>
</div>

=== "Complexity"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/progressive-complexity.png" alt="Progressive Complexity: 6 steps from a single Agent call to a full Multi-Agent System" />
    </div>

=== "Agent model"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/agent-overview.png" alt="Agent = Engine + Tools + State" />
    </div>

=== "Engine"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/engine-switchboard.png" alt="Engine Switchboard: swap LLMEngine, Plan, HumanEngine, or Supervisor" />
    </div>

=== "Recursive tools"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/recursive-tools.png" alt="Recursive Tools: functions, MCP, plans, and agents all compose through the same tool interface" />
    </div>

=== "Plan → Agent"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/plan-calls-agent.png" alt="Plan Calls Agent: a plan step can invoke a nested agent" />
    </div>

=== "Human review"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/human-review-loop.png" alt="Human Review Loop: agent produces output, human approves, edits, or stops" />
    </div>

=== "Pulse overview"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/pulse-agent-overview.jpg" alt="PulseAgent Overview: LazyPulse adds adapters, tick loop, and policy to a LazyBridge Agent" />
    </div>

=== "Tick loop"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/pulse-tick-loop.jpg" alt="One Tick, Start to Finish: Recover, Intake, Execute, Emit on every heartbeat" />
    </div>

=== "Policy"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/pulse-policy-matrix.jpg" alt="Trust &times; Action Matrix: PulsePolicy authorizes by sender identity before the worker runs" />
    </div>

</div>

<!-- ═══════════════════ CODE BLOCK ═══════════════════ -->
<div class="lb-code-panel" markdown="1">

<span class="lb-code-caption">Same mental model: from a one-line agent to nested, checkpointed pipelines.</span>

=== "Simplest"

    ```python
    from lazybridge import Agent, LLMEngine

    agent = Agent(engine=LLMEngine("claude-sonnet-4-6"))
    print(agent("Explain LazyBridge in one sentence.").text())
    ```

=== "Pipelines that nest"

    ```python
    from lazybridge import Agent, LLMEngine, Plan, Step, Session, from_step

    search    = Agent(engine=LLMEngine("gpt-5.4-mini"),      name="search")
    summarise = Agent(engine=LLMEngine("gemini-2.5-pro"),    name="summarise")
    writer    = Agent(engine=LLMEngine("claude-sonnet-4-6"), name="write")

    research = Agent(
        engine=Plan(Step("search"), Step("summarise")),
        tools=[search, summarise], name="research",
    )
    article = Agent(
        engine=Plan(Step("research"),
                    Step("write", context=from_step("research"))),
        tools=[research, writer], session=Session(),
    )
    print(article("AI agents in 2026").text())
    ```

=== "Agents route each other"

    ```python
    from lazybridge import Agent, AgentPool, LLMEngine, conclude

    pool = AgentPool()
    researcher = Agent(engine=LLMEngine("claude-sonnet-4-6", max_tool_calls_per_turn=1),
                       name="researcher", tools=[pool.as_tool(), conclude])
    writer     = Agent(engine=LLMEngine("claude-sonnet-4-6", max_tool_calls_per_turn=1),
                       name="writer", tools=[pool.as_tool(), conclude])
    pool.register(researcher, writer)        # register after construction

    # one pool = one local action space; researcher and writer self-organise
    # inside it. researcher may route("writer", ...); any agent may conclude(...)
    print(researcher("Brief me on 2026 AI trends.").text())
    ```

</div>

<!-- ═══════════════════ ECOSYSTEM ═══════════════════ -->
<section class="lb-ecosystem">
  <div class="lb-ecosystem__header">
    <h2>The LazyBridge ecosystem</h2>
    <p>Not an agent framework &mdash; a composition runtime. A stable core, its extensions, and a finance data stack &mdash; one mental model.</p>
    <p class="lb-eco-note">Only <strong>LazyBridge</strong> is published on PyPI. Everything else installs from GitHub &mdash; copy the command on each card (append <code>@vX.Y.Z</code> to pin a release tag).</p>
  </div>

  <div class="lb-ecosystem__cards">

    <a href="https://core.lazybridge.com/" class="lb-eco-card">
      <div class="lb-eco-card__badge lb-eco-card__badge--core">core</div>
      <h3>LazyBridge</h3>
      <p>The composition runtime. Build agents and pipelines where functions, plans, humans, and other agents all share one tool interface.</p>
      <div class="lb-eco-card__footer">
        <code>pip install lazybridge</code>
        <span class="lb-eco-arrow">&rarr;</span>
      </div>
    </a>

    <a href="https://pulse.lazybridge.com/" class="lb-eco-card">
      <div class="lb-eco-card__badge lb-eco-card__badge--pulse">pulse</div>
      <h3>LazyPulse</h3>
      <p>Turn workflows into always-on governed services. Gmail push notifications, Telegram, and webhooks feed a tick loop that runs policy, requests review, and keeps an audit trail.</p>
      <div class="lb-eco-card__footer">
        <code>pip install "lazypulse @ git+https://github.com/selvaz/LazyPulse.git"</code>
        <span class="lb-eco-arrow">&rarr;</span>
      </div>
    </a>

    <a href="https://tools.lazybridge.com/" class="lb-eco-card">
      <div class="lb-eco-card__badge lb-eco-card__badge--tools">tools</div>
      <h3>LazyTools</h3>
      <p>Connector clients, reusable tool providers, and safety wrappers for bringing external systems into LazyBridge without bloating the core.</p>
      <div class="lb-eco-card__footer">
        <code>pip install "lazytoolkit @ git+https://github.com/selvaz/LazyTools.git"</code>
        <span class="lb-eco-arrow">&rarr;</span>
      </div>
    </a>

    <a href="https://github.com/selvaz/LazyCrawler" class="lb-eco-card">
      <div class="lb-eco-card__badge lb-eco-card__badge--data">data</div>
      <h3>LazyCrawler</h3>
      <p>Web crawl &amp; search in three modes &mdash; pure (no-LLM), local-ML (zero tokens), and smart (LazyBridge LLM) &mdash; with a secure-by-default SSRF guard. Surfaced into agents through LazyTools.</p>
      <div class="lb-eco-card__footer">
        <code>pip install "lazycrawler @ git+https://github.com/selvaz/LazyCrawler.git"</code>
        <span class="lb-eco-arrow">&rarr;</span>
      </div>
    </a>

    <a href="https://github.com/selvaz/market-data-hub" class="lb-eco-card">
      <div class="lb-eco-card__badge lb-eco-card__badge--finance">finance</div>
      <h3>market-data-hub</h3>
      <p>The single source of financial data for the ecosystem: a DuckDB warehouse with a read-only discovery, resolution and extraction API. Agents reach it through LazyTools&rsquo; <code>datahub_*</code> tools &mdash; ids in, bounded results out.</p>
      <div class="lb-eco-card__footer">
        <code>pip install "market-data-hub @ git+https://github.com/selvaz/market-data-hub.git"</code>
        <span class="lb-eco-arrow">&rarr;</span>
      </div>
    </a>

  </div>
</section>

<!-- ═══════════════════ LAZYPULSE CASE ═══════════════════ -->
<section class="lb-pulse-case">
  <div class="lb-pulse-case__header">
    <div class="lb-eco-card__badge lb-eco-card__badge--pulse">pulse</div>
    <h2>From workflow to governed service.</h2>
    <p class="lb-pulse-case__sub">
      LazyPulse runs LazyBridge agents on inboxes, webhooks and schedules,
      applies policy before risky actions, escalates to human review
      and logs every decision.
    </p>
  </div>

  <div class="lb-pulse-flow">
    <div class="lb-pulse-step">
      <span class="lb-pulse-step__label">Input</span>
      <span class="lb-pulse-step__value">Telegram &bull; Gmail &bull; Webhook</span>
    </div>
    <span class="lb-pulse-arrow">&darr;</span>
    <div class="lb-pulse-step">
      <span class="lb-pulse-step__label">Agent</span>
      <span class="lb-pulse-step__value">LazyPulse agent &mdash; always-on, tick-driven</span>
    </div>
    <span class="lb-pulse-arrow">&darr;</span>
    <div class="lb-pulse-step">
      <span class="lb-pulse-step__label">Policy</span>
      <span class="lb-pulse-step__value">Auto-approve or escalate to human review</span>
    </div>
    <span class="lb-pulse-arrow">&darr;</span>
    <div class="lb-pulse-step">
      <span class="lb-pulse-step__label">Audit</span>
      <span class="lb-pulse-step__value">Every decision logged &mdash; tamper-proof</span>
    </div>
  </div>

  <div class="lb-pulse-case__cta">
    <a href="https://pulse.lazybridge.com/" class="lb-btn lb-btn--primary">Explore LazyPulse &rarr;</a>
  </div>
</section>

<!-- ═══════════════════ UPDATES ═══════════════════ -->
<section class="lb-updates">
  <div class="lb-updates__header">
    <h2>What's new</h2>
    <p class="lb-updates__sub">Key releases and milestones across the LazyBridge ecosystem.</p>
  </div>

  <ul class="lb-update-list">

    <li class="lb-update-item">
      <div class="lb-update-item__meta">
        <span class="lb-update-date">Jul 2026</span>
        <span class="lb-eco-card__badge lb-eco-card__badge--core">core</span>
      </div>
      <div class="lb-update-item__body">
        <p class="lb-update-title">LazyBridge 1.0.1 &mdash; the first Stable release</p>
        <p class="lb-update-desc"><code>lazybridge.__stability__</code> moves <code>"beta"</code> &rarr; <code>"stable"</code>: the core public API contract (<code>Agent</code>, <code>Plan</code>, <code>Tool</code>, <code>Envelope</code>) won't break without a major version bump. Guardrails and Checkpoint/resume are promoted Alpha &rarr; Stable in this release, backed by a live adversarial/load stress-testing pass &mdash; <code>LLMGuard</code> resisted a deliberate prompt-injection/tag-smuggling attempt, and <code>Plan</code> checkpoint/resume correctly resumed after a forced step failure without re-billing the completed step. Not everything is done: Native tools, <code>HumanEngine</code>/<code>SupervisorEngine</code>, Evals, and the Visualizer remain Alpha/Experimental, called out explicitly rather than glossed over. <a href="https://core.lazybridge.com/#maturity">Maturity table &rarr;</a> &middot; <a href="https://core.lazybridge.com/changelog/">Changelog &rarr;</a></p>
      </div>
    </li>

    <li class="lb-update-item">
      <div class="lb-update-item__meta">
        <span class="lb-update-date">Jun 2026</span>
        <span class="lb-eco-card__badge lb-eco-card__badge--pulse">pulse</span>
      </div>
      <div class="lb-update-item__body">
        <p class="lb-update-title">Event-driven Gmail &mdash; push notifications are the new default</p>
        <p class="lb-update-desc">Gmail now tells the agent when mail arrives instead of being polled: <code>GmailPushInbox</code> arms Gmail's <code>users.watch</code> onto a Cloud&nbsp;Pub/Sub topic (and re-arms it before expiry), receives push notifications on a token-protected endpoint, and makes <strong>one</strong> <code>history.list</code> call per email received &mdash; <strong>zero Gmail API calls while the mailbox is quiet</strong>. At-least-once delivery with a persisted cursor that never skips mail, burst-safe draining, self-healing resync, and exponential backoff on adapter failures. <a href="https://pulse.lazybridge.com/gmail/">Gmail push &amp; polling guide &rarr;</a></p>
      </div>
    </li>

    <li class="lb-update-item">
      <div class="lb-update-item__meta">
        <span class="lb-update-date">Jun 2026</span>
        <span class="lb-eco-card__badge lb-eco-card__badge--core">core</span>
      </div>
      <div class="lb-update-item__body">
        <p class="lb-update-title">True token streaming for Plan &amp; ReplanEngine</p>
        <p class="lb-update-desc"><code>Agent(engine=plan).stream(&hellip;)</code> now streams tokens <strong>live from each sequential step</strong> as the pipeline runs &mdash; watch it think step by step instead of waiting for one final block. Parallel bands stay silent (no token interleaving), early disconnects cancel the in-flight run, and <code>Plan(stream_buffer=)</code> bounds the queue. Plus <code>Plan.store_write_is_current()</code>: sidecar consumers can now mechanically detect the checkpoint crash-window instead of reconciling by hand. <a href="https://core.lazybridge.com/guides/full/plan/#streaming">Plan streaming &rarr;</a></p>
      </div>
    </li>

    <li class="lb-update-item">
      <div class="lb-update-item__meta">
        <span class="lb-update-date">Jun 2026</span>
        <span class="lb-eco-card__badge lb-eco-card__badge--core">core</span>
      </div>
      <div class="lb-update-item__body">
        <p class="lb-update-title">ReplanEngine &mdash; the adaptive replan loop, with checkpoint/resume</p>
        <p class="lb-update-desc">The dynamic counterpart to <code>Plan</code>: instead of a fixed DAG, a <strong>planner agent</strong> (built with <code>output=PlanRound</code>) decides the work one round at a time. <code>ReplanEngine</code> dispatches the tasks it emits &mdash; agents, plain functions, or pool routes alike &mdash; running <code>parallel=True</code> siblings concurrently, then feeds the results back for the next round. Workers can be plain functions, so it works on a simple agent too. Pass <code>store=</code> + <code>checkpoint_key=</code> and it <strong>checkpoints after every round</strong>, so a crash resumes from the right round instead of redoing completed work &mdash; same single-writer semantics as <code>Plan</code>. <a href="https://core.lazybridge.com/guides/full/replan-engine/">ReplanEngine guide &rarr;</a> &middot; <a href="https://core.lazybridge.com/recipes/dynamic-replanning/">Dynamic re-planning recipe &rarr;</a></p>
      </div>
    </li>

    <li class="lb-update-item">
      <div class="lb-update-item__meta">
        <span class="lb-update-date">Jun 2026</span>
        <span class="lb-eco-card__badge lb-eco-card__badge--tools">tools</span>
      </div>
      <div class="lb-update-item__body">
        <p class="lb-update-title">Code Support Agent &mdash; delegate to Claude Code &amp; Codex</p>
        <p class="lb-update-desc">LazyTools adds a <strong>Code Support Agent</strong> connector that puts <strong>Claude Code</strong> and <strong>Codex</strong> behind a LazyBridge agent &mdash; each in two modes: <strong>CLI</strong> (the CLI is the agent: one call, one delegated task) or <strong>MCP</strong> (the CLI exposes its own tools for your agent to orchestrate). <strong>Read-only by construction</strong>: write access lives behind the gated <code>CodeWriteTools</code> capability (mandatory <code>base_dir</code> sandbox + one-shot confirmation). <code>build_cli_collaboration()</code> runs the default three-session flow &mdash; Claude Code analyses, Codex critiques, a synthesizer writes the <em>plan</em>; execution is opt-in and sandboxed. <a href="https://tools.lazybridge.com/code-support/">See the guide &rarr;</a></p>
      </div>
    </li>

    <li class="lb-update-item">
      <div class="lb-update-item__meta">
        <span class="lb-update-date">May 2026</span>
        <span class="lb-eco-card__badge lb-eco-card__badge--core">core</span>
      </div>
      <div class="lb-update-item__body">
        <p class="lb-update-title">Pool chains &mdash; dynamic workflows from bounded local action spaces</p>
        <p class="lb-update-desc"><code>AgentPool</code> isn't just a routing registry &mdash; it's a <strong>bounded local action space</strong>: the set of specialists an agent may reach directly via a single <code>route(agent_name,&nbsp;task)</code> tool. Give a chosen agent access to a second pool and it becomes a <strong>gateway</strong> that links one local world to the next, composing a <strong>pool chain</strong>. The builder defines the local worlds and transition points; the runtime path emerges from agent decisions. Where <code>Plan</code> is the <em>static</em> workflow runtime (topology known up front), <code>AgentPool</code> is the <em>dynamic</em> one &mdash; and <code>conclude(&hellip;)</code> lets a terminal agent end the whole chain. Pair with <code>LLMEngine(max_tool_calls_per_turn=1)</code> for a single, traceable path. <a href="https://core.lazybridge.com/guides/mid/dynamic-graph/">Dynamic graph &rarr;</a> &middot; <a href="https://core.lazybridge.com/guides/mid/pool-chain/">Pool chains &rarr;</a></p>
      </div>
    </li>

    <li class="lb-update-item">
      <div class="lb-update-item__meta">
        <span class="lb-update-date">May 2026</span>
        <span class="lb-eco-card__badge lb-eco-card__badge--core">core</span>
      </div>
      <div class="lb-update-item__body">
        <p class="lb-update-title">First framework to support Claude Opus 4.8</p>
        <p class="lb-update-desc">LazyBridge ships <code>claude-opus-4-8</code> support on day one &mdash; adaptive thinking, no-sampling mode, updated thinking-token extraction (SDK &ge;&nbsp;0.105), and tier cascade (<code>top&nbsp;&rarr;&nbsp;opus-4-8</code>, <code>expensive&nbsp;&rarr;&nbsp;opus-4-7</code>).</p>
      </div>
    </li>

    <li class="lb-update-item">
      <div class="lb-update-item__meta">
        <span class="lb-update-date">May 2026</span>
        <span class="lb-eco-card__badge lb-eco-card__badge--pulse">pulse</span>
      </div>
      <div class="lb-update-item__body">
        <p class="lb-update-title">LazyPulse v0.2.0 &mdash; cron triggers &amp; PulsePolicy</p>
        <p class="lb-update-desc">Schedule agents on cron expressions with timezone support. <code>PulsePolicy</code> gates every action on sender trust level before the worker runs, with concurrency-safe compare-and-swap scheduling.</p>
      </div>
    </li>

    <li class="lb-update-item">
      <div class="lb-update-item__meta">
        <span class="lb-update-date">May 2026</span>
        <span class="lb-eco-card__badge lb-eco-card__badge--core">core</span>
      </div>
      <div class="lb-update-item__body">
        <p class="lb-update-title">LazyBridge v0.9 &mdash; one framework becomes three</p>
        <p class="lb-update-desc">v0.9 marks the point where a single monolithic package was split into three focused libraries: <strong>LazyBridge</strong> (the composition runtime), <strong>LazyPulse</strong> (always-on governed services), and <strong>LazyTools</strong> (connector clients and safety wrappers). Each package ships and versions independently, so teams only install what they need.</p>
      </div>
    </li>

  </ul>
</section>

<!-- ═══════════════════ PROVIDERS ═══════════════════ -->
<section class="lb-providers">
  <p class="lb-providers__caption">Native adapters for the leading providers &mdash; plus any model via LiteLLM or LM Studio</p>
  <div class="lb-providers__logos">
    <img src="assets/providers/anthropic.svg" alt="Anthropic" title="Anthropic" />
    <img src="assets/providers/openai.svg"    alt="OpenAI"    title="OpenAI" />
    <img src="assets/providers/google.svg"    alt="Google"    title="Google" />
    <img src="assets/providers/deepseek.svg"  alt="DeepSeek"  title="DeepSeek" />
    <img src="assets/providers/ollama.svg"    alt="Ollama"    title="Ollama" />
  </div>
</section>

<!-- Footer lives in the theme footer block (overrides/main.html) so it spans the
     full page width instead of the centred content column. -->
