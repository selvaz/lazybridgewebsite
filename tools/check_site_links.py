"""Check rendered local links and optionally the site's outgoing guide links."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen
import argparse
import re
import sys
import ssl

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("--external", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
site = root / "site"

class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids, self.links = set(), []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        field = "href" if tag == "a" else "src" if tag in ("img", "script") else None
        if field and field in attrs:
            self.links.append(attrs[field])

pages = {p.resolve(): Page(p.read_text(encoding="utf-8")) for p in site.rglob("*.html")}
failures = []
count = 0
for file, page in pages.items():
    for link in page.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        count += 1
        target = ((site / unquote(url.path).lstrip("/")) if url.path.startswith("/")
                  else file.parent / unquote(url.path)) if url.path else file
        if target.is_dir():
            target /= "index.html"
        target = target.resolve()
        if not target.exists():
            failures.append(f"{file.relative_to(site)} -> {link}")
        elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
            failures.append(f"Missing fragment: {file.relative_to(site)} -> {link}")
print(f"Rendered local links/assets/fragments checked: {count}; failures: {len(failures)}")
for failure in failures:
    print(failure)
if args.external:
    try:
        import certifi
        trust = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        trust = ssl.create_default_context()
    # Only authored pages and navigation/footer links, not theme telemetry URLs.
    texts = [p.read_text(encoding="utf-8") for p in (root / "docs").rglob("*.md")]
    texts += [(root / "overrides/main.html").read_text(encoding="utf-8")]
    # Redirects are checked against source paths separately; avoid 100+ network reads.
    texts += [(root / "mkdocs.yml").read_text(encoding="utf-8").split("nav:")[1]]
    urls = sorted(set(re.findall(r'https://(?:core|tools|pulse)\.lazybridge\.com/[^\s)"<>]*', "\n".join(texts))))

    def check(url):
        try:
            with urlopen(Request(url, headers={"User-Agent": "LazyBridge-docs-link-check"}), timeout=20, context=trust) as response:
                return url, response.status
        except Exception as error:
            return url, str(error)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(check, urls))
    print(f"Outgoing guide links checked: {len(results)}")
    for url, status in results:
        print(status, url)
    failures.extend(url for url, status in results if status != 200)
sys.exit(bool(failures))
