---
hide:
  - navigation
  - toc
title: LazyBridge — Build LLM workflows without orchestration boilerplate
description: Zero-boilerplate multi-provider LLM agent framework. Compose LLM pipelines that are also tools.
---

<!-- ═══════════════════ HERO ═══════════════════ -->
<section class="lb-hero">
  <div class="lb-hero__copy">
    <div class="lb-pill">Zero-boilerplate &bull; Multi-provider &bull; Python framework</div>

    <h1>Build LLM workflows without<br>orchestration <span class="accent">boilerplate</span></h1>

    <p class="lb-subhead">
      LazyBridge lets you compose LLMs, Python functions,
      deterministic plans, humans, memory, sessions, stores
      and external tools through one model:
      <a href="https://core.lazybridge.com/concepts/everything-is-a-tool/" class="lb-inline-link">everything is a tool.</a>
    </p>

    <div class="lb-cta-row">
      <a href="https://core.lazybridge.com/quickstart/" class="lb-btn lb-btn--primary">Get started &rarr;</a>
      <a href="https://core.lazybridge.com/concepts/layered-composition/" class="lb-btn lb-btn--ghost">Read the concepts</a>
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
    <img src="assets/hero.png" alt="LazyBridge mascot — run your pipelines" />
  </div>
</section>

<!-- ═══════════════════ FEATURE CARDS ═══════════════════ -->
<section class="lb-feature-cards">
  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>
    </div>
    <div class="lb-card__body">
      <h3>Recursive composition</h3>
      <p>A Plan is an Agent. An Agent is a Tool. Pipelines compose without glue, at any depth, with automatic cost and observability rollup.</p>
      <a href="https://core.lazybridge.com/concepts/layered-composition/" class="lb-card__link">Learn more &rarr;</a>
    </div>
  </div>

  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>
    </div>
    <div class="lb-card__body">
      <h3>Plans fail fast</h3>
      <p>PlanCompileError catches duplicate names, forward references, type drift, and broken sentinels — at construction, before any LLM call.</p>
      <a href="https://core.lazybridge.com/guides/full/plan/" class="lb-card__link">Learn more &rarr;</a>
    </div>
  </div>

  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></svg>
    </div>
    <div class="lb-card__body">
      <h3>Observable by default</h3>
      <p>Session + OpenTelemetry semconv. Cost rollup across nested agents, GenAI-standard spans, structured event log — opt-out, not opt-in.</p>
      <a href="https://core.lazybridge.com/guides/mid/session/" class="lb-card__link">Learn more &rarr;</a>
    </div>
  </div>
</section>

<!-- ═══════════════════ CODE BLOCK ═══════════════════ -->
<div class="lb-code-panel">
  <div class="lb-code-header">
    <span class="lb-code-caption">The simple case stays one line. The architecture expands only when the problem grows — without changing the mental model.</span>
    <span class="lb-code-lang">Python</span>
  </div>

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

=== "With verify + resume"

    ```python
    from lazybridge import Agent, LLMEngine, Plan, Step, Session, Store, from_step
    from lazybridge.ext.otel import OTelExporter

    judge     = Agent(engine=LLMEngine("claude-sonnet-4-6"), name="judge")
    search    = Agent(engine=LLMEngine("gpt-5.4-mini"),      name="search")
    summarise = Agent(engine=LLMEngine("gemini-2.5-pro"),    name="summarise")
    writer    = Agent(engine=LLMEngine("claude-sonnet-4-6"), name="write")

    research = Agent(
        engine=Plan(Step("search"), Step("summarise")),
        tools=[search, summarise], name="research",
    )
    article = Agent(
        engine=Plan(
            Step("research"),
            Step("write", context=from_step("research"),
                 verify=judge, max_verify=3, checkpoint_key="write"),
            checkpoint_key="research",
        ),
        tools=[research, writer],
        session=Session(store=Store(db="run.sqlite"),
                        exporters=[OTelExporter()]),
    )
    print(article("AI agents in 2026").text())
    ```

</div>

<!-- ═══════════════════ ECOSYSTEM ═══════════════════ -->
<section class="lb-ecosystem">
  <div class="lb-ecosystem__header">
    <h2>The LazyBridge ecosystem</h2>
    <p>Three focused packages. One mental model.</p>
  </div>

  <div class="lb-ecosystem__cards">

    <a href="https://core.lazybridge.com/" class="lb-eco-card">
      <div class="lb-eco-card__badge lb-eco-card__badge--core">core</div>
      <h3>LazyBridge</h3>
      <p>The agent runtime. Multi-provider orchestration through one contract — Engine + Tools + State. From a single agent to nested Plan-of-Plans.</p>
      <div class="lb-eco-card__footer">
        <code>pip install lazybridge</code>
        <span class="lb-eco-arrow">&rarr;</span>
      </div>
    </a>

    <a href="https://tools.lazybridge.com/" class="lb-eco-card">
      <div class="lb-eco-card__badge lb-eco-card__badge--tools">tools</div>
      <h3>LazyTools</h3>
      <p>Reusable tool providers, connector clients, and safety wrappers. MCP, gateways, doc-skill pipelines — ready to drop into any agent.</p>
      <div class="lb-eco-card__footer">
        <code>pip install lazytoolkit</code>
        <span class="lb-eco-arrow">&rarr;</span>
      </div>
    </a>

    <a href="https://pulse.lazybridge.com/" class="lb-eco-card">
      <div class="lb-eco-card__badge lb-eco-card__badge--pulse">pulse</div>
      <h3>LazyPulse</h3>
      <p>Always-on orchestration. Tick loops, event adapters, policy enforcement, and audit ledger for production-grade agentic services.</p>
      <div class="lb-eco-card__footer">
        <code>pip install lazypulse</code>
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
