#!/usr/bin/env python3
"""Print a Phase 0-style environment snapshot. No downloads."""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path


def main() -> None:
    free_gb = shutil.disk_usage("/").free / (1024**3)
    print(f"python={sys.version.split()[0]}")
    print(f"executable={sys.executable}")
    print(f"platform={platform.platform()}")
    print(f"machine={platform.machine()}")
    print(f"disk_free_gb={free_gb:.1f}")
    print(f"repo={Path(__file__).resolve().parents[1]}")
    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"mps_available={torch.backends.mps.is_available()}")
    except ImportError:
        print("torch=not_installed")


if __name__ == "__main__":
    main()
