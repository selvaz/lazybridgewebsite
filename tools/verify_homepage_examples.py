"""Execute site examples against read-only source checkouts, without live calls.

Usage: python -B tools/verify_homepage_examples.py --bridge PATH --tools PATH
No source install, source writes, authentication, quota reads or CLI jobs.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import os
from pathlib import Path
import re
import shlex
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("--bridge", required=True, type=Path)
parser.add_argument("--tools", required=True, type=Path)
args = parser.parse_args()
sys.path[:0] = [str(args.bridge.resolve()), str((args.tools / "src").resolve())]
site = Path(__file__).resolve().parents[1]

import lazybridge
import lazytools
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, LLMEngine
from lazybridge.engines.claude_code.protocol import ClaudeSdkResult
from lazybridge.engines.codex.app_server import CodexRunResult
from lazybridge.engines.claude_code.sdk_client import AgentSdkClient
from lazybridge.engines.codex.app_server import CodexAppServerClient
from lazytools.code_bridge.cli import build_parser
from lazytools.mcp_server.__main__ import _parse_args
from lazytools.mcp_server.providers import PROVIDER_FACTORIES

assert Path(lazybridge.__file__).resolve().is_relative_to(args.bridge.resolve())
assert Path(lazytools.__file__).resolve().is_relative_to(args.tools.resolve())
print("Imports resolve to the requested read-only checkouts.")
for transport in (AgentSdkClient, CodexAppServerClient):
    patch.object(transport, "run", side_effect=AssertionError("Unmocked runtime call forbidden")).start()

def blocks(path: Path, language: str) -> list[str]:
    return re.findall(r"```" + language + r"\n(.*?)```", path.read_text(encoding="utf-8"), re.S)

class ClaudeFake:
    def __init__(self):
        self.prompts = []

    async def run(self, prompt, *, options, attachments=()):
        self.prompts.append(prompt)
        return ClaudeSdkResult(text="draft boundary function")

class CodexFake:
    def __init__(self, verdict="APPROVED"):
        self.prompts = []
        self.verdict = verdict

    async def run(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return CodexRunResult(text=self.verdict)

python_blocks = blocks(site / "docs/index.md", "python")
assert len(python_blocks) == 3
with TemporaryDirectory(prefix="lazybridge-site-examples-") as temporary:
    # Provider credentials are fake and remain in memory. Constructors only.
    with patch.dict(os.environ, {"OPENAI_API_KEY": "mock-site-check"}):
        ns = {}
        exec(compile(python_blocks[0], "homepage:engines", "exec"), ns)
    state = ns["state"]
    for engine in [ClaudeCodeEngine(), CodexEngine(), ns["agent"].engine]:
        variant = Agent(engine=engine, **state)
        assert variant.memory is state["memory"]
        assert variant.session is state["session"]
        assert "project_status" in variant._tool_map
    print("Engine alternatives construct and retain tools, Memory and Session.")

    # Inject fake transports after real constructors, before the example call.
    tree = ast.parse(python_blocks[1])
    before = ast.Module(body=tree.body[:3], type_ignores=[])
    after = ast.Module(body=tree.body[3:], type_ignores=[])
    ns = {}
    exec(compile(before, "homepage:collaboration-setup", "exec"), ns)
    claude, codex = ClaudeFake(), CodexFake()
    ns["writer"].engine._client = claude
    ns["reviewer"].engine._client = codex
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(after, "homepage:collaboration-run", "exec"), ns)
    assert ns["result"].text() == "draft boundary function"
    assert len(claude.prompts) == len(codex.prompts) == 1
    assert "draft boundary function" in codex.prompts[0]
    # Confirm the documented exhausted-review behavior with actual engines.
    claude, codex = ClaudeFake(), CodexFake("REJECTED: boundary still wrong")
    writer = Agent(engine=ClaudeCodeEngine(client=claude),
                   verify=Agent(engine=CodexEngine(client=codex)), max_verify=3)
    assert writer("Draft the function").text() == "draft boundary function"
    assert len(claude.prompts) == len(codex.prompts) == 3
    assert "boundary still wrong" in claude.prompts[-1]
    print("Opposite-engine review executes; retry exhaustion matches the caveat.")

    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(python_blocks[2], "homepage:projects", "exec"), ns)
    from lazytools.projects import owner, records
    assert owner.get_project_owner(ns["store"], "retry-fix") == "shared"
    assert records.get_project(ns["store"], "retry-fix").status == "draft"
    assert len(ns["store"].read("blackboard:project:retry-fix")["tasks"]) == 2
    print("Project example creates a draft, shared owner, task board and default brake.")

    commands = [line for b in blocks(site / "docs/index.md", "sh") for line in b.splitlines()]
    bridge_parser = build_parser()
    for command in commands:
        argv = shlex.split(command)
        if argv[0] == "lazytools-code-bridge":
            bridge_parser.parse_args(argv[1:])
        elif argv[0] == "lazytools-mcp":
            parsed = _parse_args(argv[1:])
            assert set(parsed.providers) <= PROVIDER_FACTORIES.keys()
            assert parsed.allow_unsafe
        elif argv[:4] == ["python", "-m", "pip", "install"]:
            from packaging.requirements import Requirement
            Requirement(argv[4])
        else:
            raise AssertionError(command)
    print("Bridge/MCP commands parse against real CLI parsers; install requirements parse.")

    welcome = blocks(site / "docs/blog/posts/welcome-to-lazybridge.md", "python")[-1]
    # Force the no-key branch; no provider is constructed.
    with patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()) as output:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        exec(compile(welcome, "blog:no-key", "exec"), {})
    assert "skipping the live call" in output.getvalue()
    blog = blocks(site / "docs/blog/posts/claude-writes-codex-reviews.md", "python")
    ns = {}
    exec(compile(blog[2], "blog:boundary-function", "exec"), ns)
    with contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(blog[3], "blog:boundary-cases", "exec"), ns)
    assert output.getvalue().splitlines() == [
        "Friday 2:59pm: True", "Friday 3:00pm exactly: False",
        "Saturday: False", "Monday: True",
    ]
    print("Blog no-key branch and four deployment boundary cases pass.")
print("PASS: all homepage examples verified without live model or quota calls.")
