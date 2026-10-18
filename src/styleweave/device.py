"""Pick CUDA (Kaggle), MPS (Mac), or CPU without assuming a GPU exists."""

from __future__ import annotations

from typing import Iterable


def select_device(preference: Iterable[str] | None = None) -> str:
    order = list(preference or ("cuda", "mps", "cpu"))
    try:
        import torch
    except ImportError:
        if "cpu" in order:
            return "cpu"
        raise RuntimeError("PyTorch is not installed and CPU was not in device_preference")

    available = {
        "cuda": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
        "mps": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
        "cpu": True,
    }
    for name in order:
        key = name.lower()
        if available.get(key):
            return key
    return "cpu"


def device_details(device_name: str) -> dict[str, object]:
    details: dict[str, object] = {"device": device_name, "torch": None}
    try:
        import torch
    except ImportError:
        return details
    details["torch"] = torch.__version__
    details["cuda_available"] = torch.cuda.is_available()
    details["mps_available"] = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    if device_name == "cuda" and torch.cuda.is_available():
        details["cuda_device"] = torch.cuda.get_device_name(0)
    return details
