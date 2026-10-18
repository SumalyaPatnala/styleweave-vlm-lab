#!/usr/bin/env python3
"""Print a Phase 0-style environment snapshot. No downloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from styleweave.preflight import (  # noqa: E402
    PreflightError,
    check_disk_budget,
    collect_environment,
    detect_runtime,
)


def main() -> None:
    print(f"runtime={detect_runtime()}")
    print(f"repo={ROOT}")
    print(json.dumps(collect_environment(), indent=2))
    try:
        print(json.dumps(check_disk_budget(allow_download=False), indent=2))
    except PreflightError as exc:
        print(f"disk_preflight={exc}")


if __name__ == "__main__":
    main()
