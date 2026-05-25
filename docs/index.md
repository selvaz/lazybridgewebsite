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
    <img src="assets/hero.png" alt="LazyBridge mascot &mdash; run your pipelines" />
  </div>
</section>

<!-- ═══════════════════ FEATURE CARDS ═══════════════════ -->
<section class="lb-feature-cards">
  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>
    </div>
    <div class="lb-card__body">
      <h3>Agent frameworks create orchestration debt.</h3>
      <p>Chains, graphs, routers, executors, tools and agents become separate concepts with separate APIs. LazyBridge collapses them into one recursive model: everything is a tool.</p>
      <a href="https://core.lazybridge.com/concepts/layered-composition/" class="lb-card__link">See how it composes &rarr;</a>
    </div>
  </div>

  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>
    </div>
    <div class="lb-card__body">
      <h3>Fail before tokens burn.</h3>
      <p>Deterministic plans catch duplicate steps, broken references, type drift, and invalid sentinels before the first LLM call runs &mdash; not at 3am when your pipeline breaks in production.</p>
      <a href="https://core.lazybridge.com/guides/full/plan/" class="lb-card__link">See validation &rarr;</a>
    </div>
  </div>

  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></svg>
    </div>
    <div class="lb-card__body">
      <h3>Trace nested systems.</h3>
      <p>Agents can call plans, plans can call agents, and sub-agents can become tools &mdash; while sessions, event logs, costs, and OpenTelemetry spans roll up automatically.</p>
      <a href="https://core.lazybridge.com/guides/mid/session/" class="lb-card__link">See Session &rarr;</a>
    </div>
  </div>
</section>

<!-- ═══════════════════ PROGRESSIVE COMPLEXITY ═══════════════════ -->
<section class="lb-progressive">
  <div class="lb-progressive__header">
    <h2>Start simple. Add power without changing the model.</h2>
    <p>one model, <strong>more capability</strong></p>
  </div>
  <div class="lb-diagram-wrap">
    <img src="assets/diagrams/progressive-complexity.png" alt="Progressive Complexity: 6 steps from a single Agent call to a full Multi-Agent System" />
  </div>
</section>

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

</div>

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

<!-- ═══════════════════ CONCEPTS ═══════════════════ -->
<div class="lb-concepts" markdown="1">

<div class="lb-concepts__header">
<h2>Under the hood</h2>
<p class="lb-concepts__sub">The mental model behind the composition layer.</p>
</div>

=== "Agent model"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/agent-overview.png" alt="Agent = Engine + Tools + State: Engine decides, Tools acts, State keeps" />
    </div>

=== "Engine switchboard"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/engine-switchboard.png" alt="Engine Switchboard: swap LLMEngine, Plan, HumanEngine, or Supervisor to control autonomy" />
    </div>

=== "Recursive tools"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/recursive-tools.png" alt="Recursive Tools: functions, MCP, plans, and agents all compose through the same tool interface" />
    </div>

=== "Plan calls agent"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/plan-calls-agent.png" alt="Plan Calls Agent: a plan step can invoke a nested agent" />
    </div>

=== "Human review loop"

    <div class="lb-diagram-wrap">
      <img src="assets/diagrams/human-review-loop.png" alt="Human Review Loop: agent produces output, human approves, edits, or stops" />
    </div>

</div>

<!-- ═══════════════════ ECOSYSTEM ═══════════════════ -->
<section class="lb-ecosystem">
  <div class="lb-ecosystem__header">
    <h2>The LazyBridge ecosystem</h2>
    <p>Not an agent framework &mdash; a composition runtime. Three focused packages, one mental model.</p>
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
      <p>Turn workflows into always-on governed services. Receive events from Telegram, Gmail, or webhooks, run policy, request review, and keep an audit trail.</p>
      <div class="lb-eco-card__footer">
        <code>pip install lazypulse</code>
        <span class="lb-eco-arrow">&rarr;</span>
      </div>
    </a>

    <a href="https://tools.lazybridge.com/" class="lb-eco-card">
      <div class="lb-eco-card__badge lb-eco-card__badge--tools">tools</div>
      <h3>LazyTools</h3>
      <p>Connector clients, reusable tool providers, and safety wrappers for bringing external systems into LazyBridge without bloating the core.</p>
      <div class="lb-eco-card__footer">
        <code>pip install lazytoolkit</code>
        <span class="lb-eco-arrow">&rarr;</span>
      </div>
    </a>

  </div>
</section>

<!-- ═══════════════════ PROVIDERS ═══════════════════ -->
<section class="lb-providers">
  <p class="lb-providers__caption">Works with the best models and tools</p>
  <div class="lb-providers__logos">
    <img src="assets/providers/anthropic.svg" alt="Anthropic" title="Anthropic" />
    <img src="assets/providers/openai.svg"    alt="OpenAI"    title="OpenAI" />
    <img src="assets/providers/google.svg"    alt="Google"    title="Google" />
    <img src="assets/providers/deepseek.svg"  alt="DeepSeek"  title="DeepSeek" />
    <img src="assets/providers/meta.svg"      alt="Meta"      title="Meta" />
    <img src="assets/providers/mistral.svg"   alt="Mistral"   title="Mistral" />
    <img src="assets/providers/ollama.svg"    alt="Ollama"    title="Ollama" />
  </div>
</section>
