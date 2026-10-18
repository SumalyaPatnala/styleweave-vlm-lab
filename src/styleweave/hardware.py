"""Hardware and disk logs. No downloads."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path


def disk_free_gib(path: str | os.PathLike[str] = "/") -> float:
    return shutil.disk_usage(path).free / (1024**3)


def peak_rss_bytes() -> int | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes.
        if platform.system() == "Darwin":
            return int(usage)
        return int(usage) * 1024
    except Exception:
        return None


def hardware_log(device_name: str, extra: dict | None = None) -> dict:
    log = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "disk_free_gib": round(disk_free_gib(), 3),
        "peak_rss_bytes": peak_rss_bytes(),
        "hostname_omitted": True,
    }
    if extra:
        log.update(extra)
    log["device"] = device_name
    return log


def assert_free_disk(min_free_gib: float, path: str | os.PathLike[str] = "/") -> float:
    free = disk_free_gib(path)
    if free < min_free_gib:
        raise RuntimeError(
            f"Refusing to continue: {free:.1f} GiB free, floor is {min_free_gib:.1f} GiB"
        )
    return free


def git_commit(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text().strip()
    if ref.startswith("ref:"):
        ref_path = root / ".git" / ref.split(" ", 1)[1].strip()
        if ref_path.is_file():
            return ref_path.read_text().strip()[:40]
        return None
    return ref[:40]
