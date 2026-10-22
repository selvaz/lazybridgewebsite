"""Translate the owner's 21 Oct 2026 projects_gantt export without redrawing data.

Usage: python tools/prepare_home_gantt.py D:/lbsite-gantt/gantt-it.html
Then render docs/assets/showcase/projects-gantt-en.html to projects-gantt-en.webp.
Only presentation text and the selected project set change; SVG geometry is copied.
"""

import sys
from pathlib import Path

from bs4 import BeautifulSoup


# Keys are the original project positions and task IDs, not invented milestones.
PROJECTS = {
    1: ("LazyCommittee: committee pipeline in production", {
        36: ("StressMonitor production cutover", "Compare shadow and legacy output over several weekdays, then obtain operator approval to replace the legacy StressMonitor job. Disable the parallel check once production is healthy."),
        37: ("DalioV2 staged production cutover", "After the StressMonitor cutover is verified, compare parallel output and obtain operator approval for the DalioV2 production cutover."),
        38: ("WeeklyEarnings production cutover", "Migrate the three legacy WeeklyEarnings jobs through parallel checks and an operator-approved staged production cutover."),
        39: ("WeeklyEarnings parallel checks", "With operator approval, register three parallel checks and collect matched output before proposing the WeeklyEarnings cutover."),
        40: ("Verify first production runs", "Verify the first real Daily, Recap and Watchlist runs: successful exit, published report, stored version and any warranted notification. Report a failed run to the operator."),
        41: ("DalioV2 cutover continuation", "The earlier task closed after a command-line wiring fix. Run the remaining production comparison, then propose the actual cutover for explicit operator approval."),
    }),
    6: ("Unified macroeconomic releases database", {
        0: ("M1 Interim ECO calendar feed", "Have the ECO recap read the nightly calendar figures in read-only mode, label them unofficial web data, and register the calendar database."),
        1: ("M2 Daily consensus capture", "Capture a timestamped snapshot of the Forex Factory feed in the daily pipeline, with rate limits per provider."),
        2: ("M3 Hand-checked golden set", "Check about 30 releases against official publications, including revisions, unchanged editions, conflicts and different numeric formats."),
        3: ("M4 Contracts and database", "Define the macro releases schema for indicators, events, schedules, vintages, resolutions, consensus readings and claims; register the database and import legacy calendar data."),
        4: ("M5 Certified vertical slice", "Certify BLS, BEA, Eurostat, ONS and FRED/ALFRED adapters for US CPI, payroll revisions, successive US GDP estimates, euro-area HICP and one ONS family."),
        5: ("M6 Resolver and shadow read API", "Choose official figures by issuer and edition, log discrepancies and explain empty results while the read API runs in parallel."),
        6: ("M7 Migrate certified consumers", "Move the ECO recap and LazyCommittee macro readers to the read API one certified family at a time; resume the Weekly and Saturday recap."),
        7: ("M8 Next-day comments and briefs", "Save typed claims with verbatim source passages, attach numbers in code, and compare Luna, Haiku and Sonnet on real cases."),
        8: ("M9 Extend coverage", "Classify the catalog, add France, expand country coverage, add central-bank decisions and PMI, and run a bounded ALFRED backfill."),
        9: ("M10 Retire legacy jobs", "Retire the old macro monitoring, calendar, raw archive and regional calendar digest jobs one at a time."),
        10: ("R1 Full-text recap inputs", "Use full source text for the central-bank, geopolitical, market and economics recaps, add crawler digests and source notes, and reject only the claim with a bad citation."),
        11: ("R2 Live recap and Weekly trial", "Run the four recaps and Weekly live with Telegram delivery, fix issues, then schedule the Weekly for Saturday 24 October."),
        16: ("October release and backfill", "Release and deploy the October ecosystem update; schedule the daily job at 06:30 PT and the one-off ALFRED backfill at 22:00."),
        17: ("Indicator cards for LLMs", "Create LLM indicator cards from the approved classification and connect them to recaps and LazyCommittee."),
        18: ("Map legacy readings to catalog", "Map the 2,633 queued legacy readings to the catalog and verify the first official collection."),
        19: ("Daily sovereign yields", "Add daily market snapshots and official US, Japan, UK and Canada sovereign-yield adapters, with cross-checks and a labeled publication calendar."),
        20: ("Sovereign yields in committee", "Add daily sovereign yields to LazyCommittee fixed-income views and weekly macro reporting, then release."),
        21: ("Isolate interactive web cache", "Make web-cache isolation opt-in and enable it only for the interactive MCP configuration, never for production projects."),
        22: ("Track news session provenance", "Mark production news sessions; restrict recap and calendar readers to production sessions and classify earlier sessions from timestamps and logs, or mark them unknown."),
        23: ("Combined ecosystem release", "Release indicator cards, two-level sovereign yields, contract gates, isolated web caching and daily sovereign yields in LazyCommittee, then deploy."),
        24: ("Versioned contract checks", "Strengthen and version contract checks: each check proves its criterion, remains strictly read-only and requires a backup on deployment."),
        25: ("Cost monitor clock test", "Test the cost monitor with a relative clock because its fixed test date expired on 18 October."),
        26: ("Coupled ecosystem release", "Release the macro-data update with coupled dependency pins, promote the compatibility manifest and deploy the ecosystem to production."),
        27: ("Next tools release train", "Release contract gates and isolated web caching with coupled dependency pins after the live test on 21 October."),
        28: ("Prevent empty or stale digests", "Ensure the crawler never sends an empty file or an old session digest; exit with a no-work reason and add tests in the next crawler release."),
        29: ("Explicit MCP database creation", "Make MCP tools disclose or refuse default database creation when no environment variable is configured, rather than silently creating a second database."),
    }),
    7: ("Briefs and prompts: structured delegation", {
        0: ("B1 Brief and prompt guidance", "Add guidance for briefs, intake and prompts to the ecosystem skill and sync it to Codex."),
        1: ("B2 Establish baseline metrics", "Measure rework and failed bridge jobs per brief, plus the cached-token ratio in recap ledgers."),
        2: ("B3 Review production prompts", "Remove role preambles, place stable content first and put source declarations in the appendix."),
        3: ("B4 Validate bridge briefs", "Have the bridge flag missing brief sections and reject them in strict mode."),
        4: ("B5 Measure after two weeks", "Measure the results after two weeks and decide whether to keep or drop each intervention."),
    }),
    9: ("LazyCommittee absorbs the legacy committee", {
        0: ("A0 Complete inventory and checker", "Inventory tracked files, reachable helpers, 43 scheduled tasks, external consumers and unregistered state. Fail checks on missing rows, new tasks or unclassified references."),
        1: ("A1 Retire replaced jobs", "Keep the seven replaced jobs disabled until their replacements are proven for every due slot; export task definitions and test restoration with a fake scheduler."),
        2: ("A2 Consolidate operations", "Move runners, job matrices, database checks and live-project initialization to LazyCommittee, with one scheduled path per job and reconciled configuration."),
        3: ("A3 Recaps and report registry", "Move recaps and the report registry to LazyCommittee with one writer; test against consistent copies of all participating databases."),
        4: ("A4 Audit, health and contracts", "Move audit, job health, contract checks and MCP tools to LazyCommittee; verify exact release commits and separate Saturday slot dates."),
        5: ("A4b Comparable task definitions", "Generate comparable task definitions without mutations, preserving production names and state."),
        6: ("A5 Standalone committee release", "Tag the standalone release, pass CI or internal review, and verify a clean installation without the legacy committee. Do not promote to production."),
        7: ("A6 Staged production cutover", "Make the go/no-go decision and cut over job families progressively, with quiescence, exported task definitions and tested rollback."),
        8: ("A7 Observe the full cycle", "Observe the full cycle through Sunday 8 November and record every due occurrence."),
        9: ("A8 Conditional archive", "On the 16–17 November baseline, remove executable consumers, prove the legacy package is absent and retain a rollback kit and final archive tag."),
    }),
    10: ("Ecosystem review: verify and resolve findings", {
        0: ("R0 Storage migration and checks", "Verify the overnight storage migration, directory junctions, healthy overnight jobs and zero audit drift; delete old copies only with owner approval."),
        1: ("R1 Complete priority fixes", "Close the prepared priority fixes, release the operations and committee updates, and deploy runtimes in a window without running jobs."),
        2: ("R2 CEO priority fixes", "Verify and fix specialist human gates, deployment preflight and watchdog store selection through review; do not start the fleet."),
        3: ("R3 Enforce documented rules", "Enforce tokenized shell classification and a reviewer distinct from the writer; document or close the remaining rule findings."),
        4: ("R4 Remaining framework fixes", "Address coding-engine timeouts, tool-aware idle guards, plan checkpoint leases, lazy imports, SDK pins and lower-priority findings."),
        5: ("R5 Remaining tools fixes", "Address orphaned locks, Codex slots and quota, unused code, CI dependency skips, Windows matching and stale skills."),
        6: ("R6 Remaining committee fixes", "Address accepted exit codes, budget levels, replay lint, earnings evidence and guidance, freshness requirements and view quality."),
        7: ("R7 Legacy committee findings", "Coordinate with the absorption project: fix only what moves to LazyCommittee and classify everything else in the inventory."),
        8: ("R8 Remaining CEO fixes", "Address shell scope, prompt injection, subprocess credentials, delegated timeouts, dead code and accurate documentation."),
        9: ("R9 Brick editor fixes", "Fix four index errors and seven failing tests on a clean branch, then address repository hygiene and CI. Update the local checkout only with owner approval."),
        10: ("R10 Cross-cutting findings", "Address short-index citations and other cross-cutting findings; present dependency ranges, memory and Actions billing decisions to the owner."),
        11: ("R11 Make lessons structural", "Record lessons in ecosystem and delegation guidance: temporary test storage, explicit goals and completion criteria, live dry runs and restart behavior; close the dossier with a final outcome."),
    }),
    11: ("Ecosystem alignment: daily checks and recovery plan", {
        0: ("S1 Daily alignment check", "Check local and remote state, unpublished work, worktrees, review status, merged branches, tagged but undeployed releases and disk space; produce a plan with classified actions."),
        1: ("S2 Schedule daily summary", "Schedule a daily 22:00 PT check, corresponding to 06:00 Dublin time, from the released runtime, with a Telegram summary."),
        2: ("S3 Execute the daily plan", "Have sessions start from the daily plan, run safe and session-owned actions and bring owner decisions to the owner."),
        3: ("S4 Two-week review", "After two weeks, review blind spots, false positives and further automation such as cleaning merged worktrees."),
    }),
}


def main():
    soup = BeautifulSoup(Path(sys.argv[1]).read_text(encoding="utf-8"), "html.parser")
    soup.html["data-theme"] = "light"
    soup.title.string = "Project Gantt — English project selection"
    soup.select_one(".sub").string = "Updated 21 Oct 2026, 21:59 UTC · six real projects · hover over a task for details"
    for i, section in enumerate(list(soup.select("section"))):
        if i not in PROJECTS:
            section.decompose()
            continue
        heading, tasks = PROJECTS[i]
        section.h2.string = heading
        section.svg["aria-label"] = heading + ": planned dates and task status"
        for group in section.select("svg g"):
            task_id = int(group.title.text.split(" ")[0][1:])
            status = group.title.text.split("[", 1)[1].split("]", 1)[0]
            label, full = tasks[task_id]
            group.title.string = f"#{task_id} [{status}] {full}"
            group.select_one(".label").string = label if len(label) <= 33 else label[:32].rstrip() + "…"
    # Retain the original renderer's card layout, chart geometry and status colors.
    # Wider export avoids clipping the production project's September–November axis.
    style = soup.new_tag("style")
    style.string = "main{max-width:1640px} .sub{margin-bottom:8px}"
    soup.head.append(style)
    note = soup.new_tag("p", attrs={"class": "sub"})
    note.string = "Source: LazyTools projects_gantt · owner's board · English labels edited for clarity; planned dates, bars, statuses and counts unchanged. Dates on the chart are day/month."
    soup.main.append(note)
    output = Path("docs/assets/showcase/projects-gantt-en.html")
    # BeautifulSoup's HTML parser lowercases SVG viewBox; restore the SVG attribute.
    output.write_text(str(soup).replace("viewbox=", "viewBox="), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
