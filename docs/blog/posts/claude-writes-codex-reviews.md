---
date: 2026-08-24
---

# Claude Writes, Codex Reviews — No Second API Key Required

!!! info "Written by LazyBridge agents, reviewed by a human before publishing"
    This post came out of a LazyBridge agent pipeline — research, draft, and
    editorial pass, each a separate agent — and was approved by a human before
    it went live. More on why in the [blog's About note](../index.md).

You've got a writer agent turning out a small utility function — say, a guard that checks whether some text field is too long before you pass it downstream. It mostly works. Then one day it comes back with a version that blows up on `None`, and you spend twenty minutes tracing a crash back to a one-line bug a second pair of eyes would have caught in five seconds. So you think: fine, I'll add a second agent whose whole job is to review the first agent's output before it ships.

And then you stop mid-keystroke, doing the math. If the writer costs you some amount per call against a metered API key — a key you pay for by the token, the way `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` billing works — then a reviewer that runs on every single output, plus retries when it rejects something, could easily double or triple that bill for what's essentially a safety net. Is a nagging feeling of "this output looks fine, probably" really worth 3x the spend?

That's a legitimate thing to hesitate over. It's also the wrong question to be asking about LazyBridge's `ClaudeCodeEngine` and `CodexEngine`, because neither one touches a metered API key at all.

## What a "coding CLI" is, and why it changes the billing question

A coding CLI here means a local program you're already logged into on your machine — `claude` (the Claude Code CLI) or `codex` (the Codex CLI or desktop app) — authenticated against your existing Claude Pro/Max or ChatGPT/Codex plan, the same way you'd authenticate a browser session. No `ANTHROPIC_API_KEY`, no `OPENAI_API_KEY`, nothing that shows up as per-token usage on an API dashboard.

`ClaudeCodeEngine` drives the local `claude` CLI directly and runs requests against your Pro/Max subscription login. `CodexEngine` drives the local `codex` CLI (or the Codex desktop app, resolved via a `codex_executable()` lookup) and runs requests against your ChatGPT/Codex plan login. What changes is *which execution path* a given agent's requests travel down — subscription-authenticated CLI process vs. metered API key — not whether the requests are somehow free. They aren't. Your Claude or ChatGPT plan has its own usage limits and terms, and running two agents through two CLIs still uses two allotments of *something*. The distinction that actually matters for your monthly bill is this: neither engine adds metered per-token API spend on top of a plan you're already paying for.

## Write, review, retry — not a routing graph

The pattern LazyBridge gives you for "have a second agent check the first agent's work" is `verify=`, and the loop underneath it is called `verify_with_retry()`. In plain terms:

1. Run the writer agent on the task.
2. Hand the writer's output, plus the original task, to whatever you passed as `verify=`, and ask it a yes/no question: approved, or rejected with a reason?
3. Parse the verdict. Anything that reads as approval passes. Anything that reads as rejection — or that the parser doesn't recognize — is treated as REJECTED. The failure mode is fail-safe, not fail-open.
4. If rejected, retry. But not the reviewer's complaint in isolation — LazyBridge re-runs the *original task*, with the rejection feedback appended as extra context, so the writer gets another full attempt informed by what went wrong.
5. This repeats up to `max_verify` attempts (default 3). If the last attempt is still rejected, LazyBridge returns it anyway — you get *a* result, not an exception, and there's no flag on it telling you whether that result was ever actually approved.

The important design choice here is what `verify=` actually requires: anything with a `.run()` method — any `Agent`, running on any engine — or a plain callable. That's it. There's no separate routing graph, no special "reviewer" subclass. Wiring a Claude-writes/Codex-reviews loop is just picking two engines and passing one Agent as the `verify=` argument to another's `.run()` call.

## The scenario: a length guard

To make this concrete, here's a task any agent framework post should be able to make boring: write a function that rejects text over 4000 characters, without raising on `None` or an empty string. Framework-agnostic, no external dependencies, easy for a reviewer to check mechanically — which makes it a decent stress test for whether the write/review/retry loop actually catches something.

The subtlety is in one line:

```python
def max_length_policy(message):
    text = getattr(message, "text", None) or ""
    return len(text) <= 4000
```

Two edge cases get resolved in that single expression, and neither is obvious on a first read. `getattr(message, "text", None)` handles the case where `message.text` is `None` — you get `None` back instead of a crash. Then `or ""` handles it: `None or ""` evaluates to `""`, and if `message.text` was already an empty string, `"" or ""` is *also* `""` — an empty string is falsy, so the `or` fires there too. By the time `len(text)` runs, `text` is guaranteed to be a string, `None` and `""` have both been folded into the same safe default, and the length check never sees anything it can't handle. It's exactly the kind of one-liner a writer agent gets right about 80% of the time and gets subtly wrong the other 20% — dropping the `getattr` default, or checking `is not None` without covering the empty-string case.

Here's that function actually exercised, standalone:

```python
class Message:
    def __init__(self, text):
        self.text = text

def max_length_policy(message):
    text = getattr(message, "text", None) or ""
    return len(text) <= 4000

cases = [
    ("normal short text", Message("hello world")),
    ("text with None", Message(None)),
    ("text with empty string", Message("")),
    ("text over the limit", Message("x" * 4001)),
    ("text exactly at the limit", Message("x" * 4000)),
]

for label, msg in cases:
    print(f"{label}: {max_length_policy(msg)}")
```

## The full loop: Claude writes, Codex reviews

Now for the actual write/review/retry example, targeting that same task. This needs the `claude` and `codex` CLIs installed and logged in — both under your existing subscriptions, no API keys involved:

```python
from lazybridge import Agent
from lazybridge.engines import ClaudeCodeEngine, CodexEngine

TASK = (
    "Write a Python function `max_length_policy(message)` that returns True "
    "if the text on `message` is at most 4000 characters, and False otherwise. "
    "It must not raise if `message.text` is None or an empty string."
)

reviewer = Agent(
    name="codex-reviewer",
    engine=CodexEngine(),
)
writer = Agent(
    name="claude-writer",
    engine=ClaudeCodeEngine(model="sonnet"),
    verify=reviewer,
    max_verify=3,
)

result = writer(TASK)
print(result.text())
```

We ran this for real while writing this post. Claude's approved function, verbatim:

```python
def max_length_policy(message):
    text = getattr(message, "text", None) or ""
    return len(text) <= 4000
```

Notice what's *not* in the snippet above: no explicit prompt-passing between writer and reviewer, no state machine for retries, no manual parsing of "approved" vs "rejected" in your own code. `verify=reviewer` and `max_verify=3`, both passed when you *construct* the writer `Agent` — not arguments to the call itself — are the whole wiring. Under the hood, `verify_with_retry()` asks the reviewer something close to *"Evaluate this output: \<the writer's code\>. Original task: \<the task above\>. Approved or rejected, with reason?"* — you don't have to build that prompt yourself, and there's no separate object to track attempt counts or the final verdict; `result` is the same kind of value `writer(TASK)` always returns, whether or not `verify=` was ever set.

## Where this actually breaks

Three things worth knowing before you rely on this pattern.

**The retry budget is small and fixed, and LazyBridge won't tell you if it ran out.** `max_verify` defaults to 3. If your writer and reviewer are stuck in a loop — writer keeps making the same mistake, reviewer keeps rejecting it the same way — you get three attempts and then whatever the third one produced, rejected or not, with no flag on the result marking which happened. If you need to know, have your reviewer's approval show up in its own output (a trailing `APPROVED`/`REJECTED` line you check yourself, for instance) — don't assume a returned result was ever actually approved.

**Subscription plans are bounded too.** Routing through `ClaudeCodeEngine` and `CodexEngine` means you're not paying per-token on top of a plan, but Pro/Max and ChatGPT/Codex plans still have their own usage limits and terms. A write/review loop with several retries on every call is still meaningfully more usage than a single writer call — just usage against a subscription ceiling instead of a token-metered invoice.

**Approval is not correctness.** A reviewer agent approving a writer agent's output means two language models agreed on something, not that the something is right. For a task as narrow as a length guard, that's a reasonably strong signal. For anything with more room to be subtly, plausibly wrong, an approved result out of `verify_with_retry()` is a reason to trust the code a bit more than an unreviewed first draft. It's not a substitute for reading it.
