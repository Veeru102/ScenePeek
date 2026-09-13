#!/usr/bin/env python3
"""Fetch openly licensed videos listed in eval/sources.yaml with yt-dlp (optional; needs network).

    uv run --with yt-dlp python scripts/download_eval_videos.py
"""

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "eval/sources.yaml"
OUT = ROOT / "eval/videos"


def main() -> None:
    if not SOURCES.exists():
        sys.exit(f"{SOURCES} not found — list {{key, title, url, license}} entries there")
    OUT.mkdir(parents=True, exist_ok=True)
    for v in yaml.safe_load(SOURCES.read_text()).get("videos", []):
        target = OUT / f"{v['key']}.mp4"
        if target.exists():
            print("exists", target)
            continue
        print("downloading", v["title"], v["url"])
        subprocess.run(
            ["yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]", "--merge-output-format", "mp4",
             "-o", str(target), v["url"]],
            check=True,
        )


if __name__ == "__main__":
    main()
