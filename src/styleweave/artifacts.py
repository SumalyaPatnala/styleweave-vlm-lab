"""Keep metrics.json from a failed or in-progress run from looking official."""

from __future__ import annotations

import json
import time
from pathlib import Path

METRICS_NAME = "metrics.json"
STATUS_NAME = "run_status.json"


def invalidate_metrics(artifacts_dir: Path, reason: str) -> Path | None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = artifacts_dir / METRICS_NAME
    moved: Path | None = None
    if metrics.is_file():
        stamp = time.strftime("%Y%m%dT%H%M%S")
        moved = artifacts_dir / f"metrics.json.invalidated-{stamp}"
        metrics.replace(moved)
    status = {
        "status": "invalidated" if moved else "no_metrics",
        "reason": reason,
        "invalidated_metrics": str(moved) if moved else None,
        "metrics_json_present": False,
    }
    (artifacts_dir / STATUS_NAME).write_text(json.dumps(status, indent=2) + "\n")
    return moved


def mark_run(artifacts_dir: Path, status: str, extra: dict | None = None) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "metrics_json_present": (artifacts_dir / METRICS_NAME).is_file()}
    if extra:
        payload.update(extra)
    (artifacts_dir / STATUS_NAME).write_text(json.dumps(payload, indent=2) + "\n")
