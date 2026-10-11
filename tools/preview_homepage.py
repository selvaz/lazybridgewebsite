"""Capture and check the local MkDocs build in headless Microsoft Edge."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "_drafts/preview"
OUTPUT.mkdir(parents=True, exist_ok=True)

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "site")))
Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_port}/"
results = []
try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, timeout=60_000)
        for label, width, height in [("desktop-1440", 1440, 1000), ("mobile-390", 390, 844)]:
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            page.set_default_timeout(60_000)
            page.set_default_navigation_timeout(60_000)
            page.goto(url, wait_until="load", timeout=60_000)
            page.evaluate("""async () => {
              document.querySelectorAll('img').forEach(img => img.loading = 'eager');
              await Promise.all([...document.images].map(img => img.decode()));
              await document.fonts.ready;
            }""")
            metrics = page.evaluate("""() => ({
              width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
              images: [...document.querySelectorAll('article img')].map(img => ({
                alt: img.alt, loaded: img.complete && img.naturalWidth > 0
              })),
              privateLabels: document.querySelectorAll('.lb-private').length,
              contacts: document.querySelectorAll('.lb-project-contact a[href="mailto:lazybridgelabs@gmail.com"]').length,
              benchmarkNotes: document.body.innerText.split('Benchmark in progress').length - 1,
              proseWords: [...document.querySelectorAll('.lb-manifesto, .lb-demo p, .lb-demo h2, .lb-showcase > p, .lb-project > p, .lb-project h3')]
                .map(el => el.innerText).join(' ').split(/\\s+/).length
            })""")
            assert metrics["scrollWidth"] <= width, metrics
            assert metrics["privateLabels"] == metrics["contacts"] == 3, metrics
            assert metrics["benchmarkNotes"] == 1, metrics
            assert all(img["alt"] and img["loaded"] for img in metrics["images"]), metrics
            page.screenshot(path=str(OUTPUT / f"home-{label}.png"), full_page=True, timeout=60_000)
            page.screenshot(path=str(OUTPUT / f"hero-{label}.png"), timeout=60_000)
            page.locator(".lb-demo").nth(2).screenshot(path=str(OUTPUT / f"bridge-{label}.png"), timeout=60_000,
                style=".md-header, .md-tabs { visibility: hidden !important; }")
            for ident in ["lazybricks", "lazycommittee", "lazyceo"]:
                page.locator(f"#{ident}").screenshot(path=str(OUTPUT / f"{ident}-{label}.png"), timeout=60_000,
                    style=".md-header, .md-tabs { visibility: hidden !important; }")
            results.append({"viewport": label, **metrics})
            print(f"PASS {label}: no horizontal overflow; all gallery images loaded; 3 private projects and contacts.")
            page.close()
        browser.close()
finally:
    server.shutdown()
(OUTPUT / "checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("Saved desktop and mobile previews to _drafts/preview/.")
