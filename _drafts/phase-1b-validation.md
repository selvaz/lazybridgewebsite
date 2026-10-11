# Phase 1b validation — 11 October 2026

- Homepage: four manifesto principles, six technical bands, three private project galleries, contact CTA and pinned installation commands.
- Prose count from the rendered homepage: 516 words excluding code and captions, measured by `tools/preview_homepage.py`.
- Source examples: PASS against LazyBridge and LazyTools read-only checkouts, with fake transports and no live model or quota calls.
- Rendered local links/assets/fragments: 575 checked, zero failures.
- Edge: Playwright `channel="msedge"`, headless, loopback HTTP server, explicit 60-second timeouts. At both 1440 and 390 pixels, no document overflow; all images loaded and have alt text; three private labels and three project contact links; exactly one benchmark note.
- Visual review: full desktop/mobile pages, heroes, bridge band and galleries inspected. Mobile hero spacing, image sizing, caption contrast and command wrapping adjusted. Gallery element captures hide the sticky header during capture; full-page captures preserve the normal site UI.
- Private repository code links/install instructions: none. Committee images contain dated, sourced statistics and an allocation hierarchy, without performance claims.
- CEO images: real GUI demo, Gantt and Program rectangular excerpts, Brake page with quota history. Captions identify demo data. No placeholder is needed because the supplied demo captures exist.
- Originals and derivatives live in `docs/assets/showcase/`; previews are in `_drafts/preview/` and remain outside the published site.
- Existing Python build dependencies match the repo requirements; no dependency installation or global changes needed.

Final command: `C:/ProgramData/spyder-6/python.exe -m mkdocs build --strict`, exit 0. Exact last lines:

```text
INFO    -  Cleaning site directory
INFO    -  Building documentation to directory: C:\Users\Administrator\Documents\GitHub\lazybridgewebsite-refresh\site
INFO    -  Documentation built in 1.98 seconds
```

The remaining owner decision is the public fixture, task set and evidence owner for the planned tandem/routing benchmark. Package maturity and later Gantt label translation follow the owner's answers.
