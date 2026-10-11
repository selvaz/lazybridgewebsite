"""Prepare web images from owner-supplied renders and real GUI screenshots.

No retouching: CEO excerpts are rectangular crops. Keep source repositories read only.
"""
from pathlib import Path
import argparse
import shutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/showcase"
parser = argparse.ArgumentParser()
parser.add_argument("--ceo", type=Path, required=True)
args = parser.parse_args()
for source in args.ceo.glob("*.png"):
    with Image.open(source) as img:
        print(source.name, img.size)

# Coordinates expressed as fractions keep the crops tied to the supplied GUI capture.
# Gantt: the lower two projects. Program: completed tasks 0–10, without the
# unrelated remediation status and missing-date task in this synthetic demo.
for name, filename, bounds in [
    ("gantt.png", "ceo-gantt.png", (0.075, 0.644, 0.925, 0.994)),
    ("program-lazycommittee.png", "ceo-program.png", (0.078, 0.134, 0.699, 0.495)),
    ("gantt.png", "ceo-gantt-detail.png", (0.075, 0.833, 0.925, 0.990)),
    ("brake.png", "ceo-brake-history.png", (0.077, 0.162, 0.701, 0.640)),
]:
    with Image.open(args.ceo / name) as img:
        w, h = img.size
        img.crop(tuple(round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(bounds))).save(ASSETS / filename)
shutil.copyfile(args.ceo / "brake.png", ASSETS / "ceo-brake.png")

for source in sorted(ASSETS.glob("*.png")):
    if source.name == "ceo-brake-meters.png":
        continue  # superseded by the real GUI demo's history chart
    with Image.open(source) as img:
        img = img.convert("RGB")
        img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        target = source.with_name(source.stem.replace("-4K", "") + ".webp")
        img.save(target, "WEBP", quality=90, method=6)
        print(f"{target.name}: {img.size}, {target.stat().st_size:,} bytes")
