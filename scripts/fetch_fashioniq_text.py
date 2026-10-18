#!/usr/bin/env python3
"""Download official FashionIQ captions + splits only (~7 MB). No images."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master"
CATEGORIES = ("dress", "shirt", "toptee")
SPLITS = ("train", "val", "test")


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        dest.write_bytes(response.read())


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "fashion-iq"
    print(f"Writing official text to {out}")
    for category in CATEGORIES:
        for split in SPLITS:
            cap = f"captions/cap.{category}.{split}.json"
            split_name = f"image_splits/split.{category}.{split}.json"
            fetch(f"{BASE}/{cap}", out / cap)
            fetch(f"{BASE}/{split_name}", out / split_name)
            print("fetched", cap, split_name)
    print("Done. Images were not downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
