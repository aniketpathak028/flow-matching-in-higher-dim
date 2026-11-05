"""Refresh the presentation website in docs/ from the latest results.

    python -m experiments.site                # reads results/, writes docs/
    python -m experiments.site --out results_x

What it does:
* results/summary.json      -> docs/data.js   (as `window.DATA = {...};`, read by docs/charts.js)
* results/videos/<v>.mp4    -> docs/videos/   (every entry of plan.VIDEOS)

docs/index.html (the text, charts and conclusions) is written by hand; this
script only refreshes the numbers and videos it shows. Run `python -m
experiments.report` first so summary.json is up to date.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from experiments.plan import video_jobs

SITE = Path("docs")


def build_site(out_root: Path, site: Path = SITE) -> None:
    summary = out_root / "summary.json"
    if not summary.exists():
        raise SystemExit(f"{summary} not found: run `python -m experiments.report --out {out_root}` first.")
    site.mkdir(parents=True, exist_ok=True)
    (site / "data.js").write_text(f"window.DATA = {summary.read_text()};\n")

    (site / "videos").mkdir(exist_ok=True)
    missing = []
    jobs = video_jobs()
    for _, _, file in jobs:
        video = out_root / "videos" / file
        if video.exists():
            shutil.copy2(video, site / "videos" / file)
        else:
            missing.append(file)
    print(f"Wrote {site / 'data.js'} and {len(jobs) - len(missing)} videos to {site / 'videos'}.")
    if missing:
        print(f"Missing videos (run `python -m experiments.run` to render them): {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results", help="results folder to read")
    parser.add_argument("--site", default=str(SITE), help="website folder to write")
    args = parser.parse_args()
    build_site(Path(args.out), Path(args.site))


if __name__ == "__main__":
    main()
