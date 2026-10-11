---
date: 2026-08-24
---

# Claude Writes, Codex Reviews — No Second API Key Required

!!! info "Written by LazyBridge agents, reviewed by a human before publishing"
    This post came out of a LazyBridge agent pipeline — research, draft, and
    editorial pass, each a separate agent — and was approved by a human before
    it went live. More on why in the [blog's About note](../index.md).

You've got a writer agent turning out a small utility function — say, one that decides whether it's actually safe to deploy right now. It mostly works. Then one day someone points out it waves a green light at 3:00pm on the dot on a Friday, because the boundary check used `>` instead of `>=`, and you spend twenty minutes tracing a very avoidable incident back to a one-character bug a second pair of eyes would have caught in five seconds. So you think: fine, I'll add a second agent whose whole job is to review the first agent's output before it ships.

And then you stop mid-keystroke, doing the math. If the writer uses a metered API key, a reviewer on every output — plus retries after rejection — adds billed calls. How much that costs depends on the task, models and retry count.

LazyBridge's `ClaudeCodeEngine` and `CodexEngine` provide another execution path: the local coding runtime's subscription login. The configured login determines the billing path; the engines do not force subscription billing if the runtime is configured for API-key authentication.

## What a "coding CLI" is, and why it changes the billing question

A coding CLI here means a local program you're already logged into on your machine — `claude` (the Claude Code CLI) or `codex` (the Codex CLI or desktop app) — authenticated against your existing Claude Pro/Max or ChatGPT/Codex plan, the same way you'd authenticate a browser session. No `ANTHROPIC_API_KEY`, no `OPENAI_API_KEY`, nothing that shows up as per-token usage on an API dashboard.

`ClaudeCodeEngine` uses the Claude Agent SDK and its local Claude Code runtime. `CodexEngine` speaks JSON-RPC to `codex app-server`, resolved through `codex_executable()`. When those runtimes are signed in under subscriptions, the calls use their plan quotas instead of per-token API billing. The calls still consume quota.

## Write, review, retry — not a routing graph

The pattern LazyBridge gives you for "have a second agent check the first agent's work" is `verify=`, and the loop underneath it is called `verify_with_retry()`. In plain terms:

1. Run the writer agent on the task.
2. Hand the writer's output, plus the original task, to whatever you passed as `verify=`, and ask it a yes/no question: approved, or rejected with a reason?
3. Parse the verdict. Anything that reads as approval passes. Anything that reads as rejection — or that the parser doesn't recognize — is treated as REJECTED. The failure mode is fail-safe, not fail-open.
4. If rejected, retry. But not the reviewer's complaint in isolation — LazyBridge re-runs the *original task*, with the rejection feedback appended as extra context, so the writer gets another full attempt informed by what went wrong.
5. This repeats up to `max_verify` attempts (default 3). If the last attempt is still rejected, LazyBridge returns it anyway — you get *a* result, not an exception (more on what that means for you below).

The important design choice is what `verify=` accepts: an `Agent` on any engine, or a plain callable. Wiring this loop means passing the reviewer as `verify=` when constructing the writer Agent, as in the example below.

## The scenario: should I deploy right now?

To make this concrete — and because not every worked example needs to be a boring input-sanitizer — here's a task every developer has an opinion about: write `should_i_deploy(now)`, which looks at the current time and returns whether it's safe to push to production, plus a message explaining why or why not. The rule: blocked from 3:00pm onward on a Friday, blocked all day on the weekend, fine otherwise.

The subtlety is in exactly where "3:00pm onward" starts:

```python
friday_cutoff = datetime.time(15, 0, 0)
is_friday_afternoon = weekday == 4 and now.time() >= friday_cutoff
```

That `>=` is doing real work. Write `>` instead, by a single-character slip, and 3:00pm precisely stops being blocked — the one moment the rule exists to name explicitly turns into the one moment it silently lets through. It's the classic shape of a boundary bug: everything either side of the line behaves correctly, so a handful of obvious manual checks (2:00pm, 4:00pm, Saturday) all pass, and the actual mistake only shows up if someone happens to test the exact second the rule changes. That's precisely the kind of thing a bored human reviewer skims past and a second agent, asked specifically to check the logic rather than skim the vibe, is well-suited to catch.

## The full loop: Claude writes, Codex reviews

Now for the actual write/review/retry example, targeting that same task. This needs the `claude` and `codex` CLIs installed and logged in — both under your existing subscriptions, no API keys involved:

```python
from lazybridge import Agent
from lazybridge.engines import ClaudeCodeEngine, CodexEngine

TASK = (
    "Write a Python function `should_i_deploy(now)` where `now` is a "
    "datetime.datetime. Return a tuple (bool, str): False with a witty, "
    "dry, deadpan-sysadmin-humor excuse if `now` falls on a Friday after "
    "15:00, or anytime on Saturday or Sunday -- true with a short "
    "encouraging message otherwise. Handle the exact Friday-15:00 boundary "
    "correctly (15:00:00 itself should already count as blocked)."
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

We ran this for real while writing this post. Claude's approved function, verbatim — including the excuse bank, because withholding it would be a crime against the bit:

```python
import datetime
import random


def should_i_deploy(now: datetime.datetime) -> tuple[bool, str]:
    weekday = now.weekday()  # Monday=0 ... Sunday=6
    friday_cutoff = datetime.time(15, 0, 0)

    is_friday_afternoon = weekday == 4 and now.time() >= friday_cutoff
    is_weekend = weekday in (5, 6)

    if is_friday_afternoon:
        excuses = [
            "It's after 3pm on a Friday. That's not a deploy window, "
            "that's a hostage situation waiting to happen.",
            "Congratulations, you've entered the Bermuda Triangle of "
            "deploy windows: Friday afternoon. Ships (and services) "
            "that enter do not come back.",
            "Sure, ship it now, and spend your weekend explaining to "
            "your family why your phone won't stop buzzing.",
            "Friday-afternoon deploys are how legends are made. "
            "Cautionary legends. Told at new-hire orientation.",
            "The on-call engineer has requested, via a strongly worded "
            "note taped to their monitor, that you not do this.",
        ]
        return False, random.choice(excuses)

    if is_weekend:
        excuses = [
            "It's the weekend. Even your CI/CD pipeline deserves a "
            "day off, and frankly, so do you.",
            "Deploying on a weekend is like performing surgery in a "
            "dark alley: technically possible, deeply inadvisable.",
            "The servers are resting. The on-call engineer is resting. "
            "Nature is healing. Please do not disturb.",
            "Weekend deploys have a 100% success rate right up until "
            "they don't, and then it's just you, alone, at 2am.",
            "This is a weekend. Weekends were invented specifically "
            "so we wouldn't push to prod today.",
        ]
        return False, random.choice(excuses)

    encouragements = [
        "Green across the board. Ship it like you mean it.",
        "The stars, the CI pipeline, and your coffee are all aligned. Go for it.",
        "No red flags, no weekend curses in effect. Deploy away.",
        "It's a perfectly reasonable time to push. Go be a hero.",
        "All clear! May your rollout be smooth and your logs be boring.",
    ]
    return True, random.choice(encouragements)
```

And yes, it actually gets the boundary right:

```python
for label, when in [
    ("Friday 2:59pm", datetime.datetime(2026, 8, 21, 14, 59, 0)),
    ("Friday 3:00pm exactly", datetime.datetime(2026, 8, 21, 15, 0, 0)),
    ("Saturday", datetime.datetime(2026, 8, 22, 10, 0, 0)),
    ("Monday", datetime.datetime(2026, 8, 24, 11, 0, 0)),
]:
    print(f"{label}: {should_i_deploy(when)[0]}")

# Friday 2:59pm: True
# Friday 3:00pm exactly: False
# Saturday: False
# Monday: True
```

Notice what's *not* in the snippet above: no explicit prompt-passing between writer and reviewer, no state machine for retries, no manual parsing of "approved" vs "rejected" in your own code. `verify=reviewer` and `max_verify=3`, both passed when you *construct* the writer `Agent` — not arguments to the call itself — are the whole wiring. Under the hood, `verify_with_retry()` asks the reviewer something close to *"Evaluate this output: \<the writer's code\>. Original task: \<the task above\>. Approved or rejected, with reason?"* — you don't have to build that prompt yourself, and there's no separate object to track attempt counts or the final verdict; `result` is the same kind of value `writer(TASK)` always returns, whether or not `verify=` was ever set.

## Where this actually breaks

Three things worth knowing before you rely on this pattern.

**The retry budget is small and fixed, and LazyBridge won't tell you if it ran out.** `max_verify` defaults to 3. If your writer and reviewer are stuck in a loop — writer keeps making the same mistake, reviewer keeps rejecting it the same way — you get three attempts and then whatever the third one produced, rejected or not, with no flag on the result marking which happened. If you need to know, have your reviewer's approval show up in its own output (a trailing `APPROVED`/`REJECTED` line you check yourself, for instance) — don't assume a returned result was ever actually approved.

**Subscription plans are bounded too.** With subscription logins, a write/review loop consumes both engines' quotas. API-key authentication follows the runtime's API billing path instead.

**Approval is not correctness.** A reviewer accepting an output is a review result. Check the boundary cases with code before using the function.

## October update: async repository work

This example checks a generated output. LazyTools 0.10.1 also provides an async code bridge for longer repository jobs: detached launches, approval tickets, cancellation and workspace evidence on failure. Tier-based `--review-of JOB_ID` selects the opposite engine. Named sessions keep native conversations durable; the shared project layer adds contracts and explicit verification records. See [collaboration on the homepage](../../index.md#collaboration) and the [bridge guide](https://tools.lazybridge.com/code-bridge/).
