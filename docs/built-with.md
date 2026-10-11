---
title: Built with LazyBridge
description: Two private projects using LazyBridge and MCP tools in document pipelines and a shared 3D editor.
---

# Built with LazyBridge

<div class="lb-home-section" markdown="1">

These two private repositories show where the engine and tool model is being used.

## LazyCommittee — private repository

LazyCommittee is an AI investment committee run as a linear pipeline of dated documents:

1. Daily digests covering geopolitics, economics, markets and central banks.
2. Weekly world state, split into strategic and tactical documents.
3. Views by asset class.
4. A portfolio document.

The pipeline uses LazyBridge with Claude Code and Codex as engines. Code computes numeric results and gives the LLM the computed outputs. Deterministic publication checks cover freshness, numeric lint and source mapping. Blocking findings gate publication; some narrative numeric lint findings remain warnings. Daily and weekly runs are scheduled.

The document registry is append-only, admits one document per name in a batch, and moves document heads through compare-and-swap. Those rules preserve document history and make concurrent publication conflicts explicit.

These projects are private. If you are interested in collaborating, write to [lazybridgelabs@gmail.com](mailto:lazybridgelabs@gmail.com).

## LazyBricks — private repository

LazyBricks is a local 3D editor for building brick models. It combines a Three.js front end with FastAPI and SQLite and indexes the full installed LDraw Official Parts Library. Search, categories, pagination, 3D thumbnails and inventory filters make the catalogue usable without loading it all into the browser.

The editor and MCP server work on the same projects and revisions, so an AI agent can propose changes while a person works in the editor. A phone interface supports touch selection, camera controls and part editing. Mobile checks have used viewport/touch emulation on a PC; a physical-phone test remains open.

Static checks cover placement, collisions, inventory and declared connections. They do not certify mechanical stability or simulate gravity, clutch forces or flexible parts. Indexing the installed official library does not imply coverage of every brick ever produced or every possible connection.

These projects are private. If you are interested in collaborating, write to [lazybridgelabs@gmail.com](mailto:lazybridgelabs@gmail.com).

[Explore engines and collaboration](index.md#engines)

</div>
