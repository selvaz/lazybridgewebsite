---
hide:
  - navigation
  - toc
title: LazyBridge
description: Zero-boilerplate multi-provider LLM agent framework. Compose LLM pipelines that are also tools.
---

<section class="lb-hero">
  <div class="lb-hero__copy">
    <div class="lb-pill">Recursive · Validated · Observable</div>
    <h1>Compose LLM pipelines<br>that are also <span class="accent">tools</span>.</h1>
    <p class="lb-subhead">LazyBridge composes LLMs, deterministic plans, humans, and external tools through one contract. Pipelines are tools. Tools are tools. Nesting has no special syntax — and every level validates at construction, before any LLM call.</p>
    <div class="lb-cta-row">
      <a href="https://core.lazybridge.com/quickstart/" class="lb-btn lb-btn--primary">Get started</a>
      <a href="https://core.lazybridge.com/concepts/layered-composition/" class="lb-btn lb-btn--ghost">How composition works →</a>
    </div>
    <div class="lb-chips">
      <span class="lb-chip">Multi-dimensional composition</span>
      <span class="lb-chip">Plan validated at construction</span>
      <span class="lb-chip">Provider freedom</span>
      <span class="lb-chip">Output validated + verify</span>
      <span class="lb-chip">Observable by default</span>
    </div>
  </div>
  <div class="lb-hero__art">
    <img src="assets/hero.png" alt="LazyBridge — compose pipelines that are tools" width="480" />
  </div>
</section>

<section class="lb-feature-cards">
  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>
    </div>
    <h3>Recursive composition</h3>
    <p>A Plan is an Agent. An Agent is a Tool. Pipelines compose without glue, at any depth, with automatic cost and observability rollup.</p>
    <a href="https://core.lazybridge.com/concepts/layered-composition/" class="lb-card__link">Learn more →</a>
  </div>
  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>
    </div>
    <h3>Plans fail fast</h3>
    <p>PlanCompileError catches duplicate names, forward references, type drift, and broken sentinels — at construction, before any LLM call.</p>
    <a href="https://core.lazybridge.com/guides/full/plan/" class="lb-card__link">Learn more →</a>
  </div>
  <div class="lb-card">
    <div class="lb-card__icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></svg>
    </div>
    <h3>Observable by default</h3>
    <p>Session + OpenTelemetry semconv. Cost rollup across nested agents, GenAI-standard spans, structured event log — opt-out, not opt-in.</p>
    <a href="https://core.lazybridge.com/guides/mid/session/" class="lb-card__link">Learn more →</a>
  </div>
</section>

<div class="lb-code-panel">
<div class="lb-code-caption">The simple case stays one line. The architecture grows only when the problem grows — without changing the mental model.</div>

=== "Pipelines that nest"

    ```python
    from lazybridge import Agent, LLMEngine, Plan, Step, Session, from_step

    search    = Agent(engine=LLMEngine("gpt-5.4-mini"),      name="search")
    summarise = Agent(engine=LLMEngine("gemini-2.5-pro"),    name="summarise")
    writer    = Agent(engine=LLMEngine("claude-sonnet-4-6"), name="write")

    research = Agent(
        engine=Plan(Step("search"), Step("summarise")),        # a sub-pipeline
        tools=[search, summarise], name="research",
    )
    article = Agent(
        engine=Plan(Step("research"),                           # research is one tool
                    Step("write", context=from_step("research"))),
        tools=[research, writer], session=Session(),
    )
    print(article("AI agents in 2026").text())
    ```

=== "Simplest"

    ```python
    from lazybridge import Agent, LLMEngine

    agent = Agent(engine=LLMEngine("claude-sonnet-4-6"))
    print(agent("Explain LazyBridge in one sentence.").text())
    ```

=== "With verify + resume"

    ```python
    from lazybridge import Agent, LLMEngine, Plan, Step, Session, Store, from_step
    from lazybridge.ext.otel import OTelExporter

    judge = Agent(engine=LLMEngine("claude-sonnet-4-6"), name="judge")

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

<section class="lb-providers">
  <p class="lb-providers__caption">Works with the best models and tools</p>
  <div class="lb-providers__logos">
    <img src="assets/providers/anthropic.svg" alt="Anthropic" />
    <img src="assets/providers/openai.svg" alt="OpenAI" />
    <img src="assets/providers/google.svg" alt="Google" />
    <img src="assets/providers/deepseek.svg" alt="DeepSeek" />
    <img src="assets/providers/meta.svg" alt="Meta" />
    <img src="assets/providers/mistral.svg" alt="Mistral" />
    <img src="assets/providers/ollama.svg" alt="Ollama" />
  </div>
</section>
