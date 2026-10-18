"""Environment-aware preflight for B0. No Qwen. No image uploads."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

# Phase 0 HTTP HEAD of openai/clip-vit-base-patch32 pytorch_model.bin
CLIP_WEIGHTS_BYTES = 605_247_071
HF_TEMP_FACTOR = 0.40
CLIP_PEAK_BYTES = int(CLIP_WEIGHTS_BYTES * (1.0 + HF_TEMP_FACTOR))
ARTIFACTS_HEADROOM_BYTES = 512 * 1024 * 1024  # embeddings + reports
MAC_MIN_FREE_GIB = 25.0
TRANSFORMERS_MIN = (4, 41, 0)
TRANSFORMERS_MAX_EXCLUSIVE = (5, 3, 0)


class PreflightError(RuntimeError):
    pass


def detect_runtime() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    if platform.system() == "Darwin":
        return "macos"
    return "other"


def default_cache_dir(runtime: str | None = None, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    if env:
        return Path(env)
    if (runtime or detect_runtime()) == "kaggle":
        return Path("/kaggle/working/hf-cache")
    return Path.home() / ".cache" / "huggingface"


def disk_free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def _parse_version(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in text.split(".")[:3]:
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def clip_weights_cached(model_name: str, cache_dir: Path) -> bool:
    slug = "models--" + model_name.replace("/", "--")
    hub = cache_dir / "hub" / slug
    snapshots = cache_dir / "hub" / slug / "snapshots"
    if snapshots.is_dir() and any(snapshots.rglob("pytorch_model.bin")):
        return True
    if hub.is_dir() and any(hub.rglob("pytorch_model.bin")):
        return True
    return False


def clip_download_budget(*, cached: bool) -> dict:
    peak = 0 if cached else CLIP_PEAK_BYTES
    required = peak + ARTIFACTS_HEADROOM_BYTES
    return {
        "clip_weights_bytes": CLIP_WEIGHTS_BYTES,
        "hf_temp_factor": HF_TEMP_FACTOR,
        "clip_peak_bytes": peak,
        "artifacts_headroom_bytes": ARTIFACTS_HEADROOM_BYTES,
        "required_free_bytes": required,
        "weights_cached": cached,
    }


def check_disk_budget(
    *,
    runtime: str | None = None,
    cache_dir: str | Path | None = None,
    allow_download: bool = False,
    model_name: str = "openai/clip-vit-base-patch32",
    mac_min_free_gib: float = MAC_MIN_FREE_GIB,
) -> dict:
    runtime = runtime or detect_runtime()
    cache = default_cache_dir(runtime, str(cache_dir) if cache_dir else None)
    free = disk_free_bytes(cache)
    cached = clip_weights_cached(model_name, cache)
    budget = clip_download_budget(cached=cached)
    errors: list[str] = []

    if runtime == "macos":
        floor = int(mac_min_free_gib * (1024**3))
        if free < floor:
            errors.append(
                f"macOS free-space floor: {free / 1024**3:.2f} GiB < {mac_min_free_gib:.1f} GiB"
            )
        if allow_download and not cached and free < budget["required_free_bytes"] + floor:
            errors.append(
                "macOS download would not leave the 25 GiB floor after CLIP peak+temp"
            )
    else:
        need = budget["required_free_bytes"]
        if not allow_download and not cached:
            errors.append(
                f"CLIP weights are not in {cache}; pass allow_download after a disk check"
            )
        if allow_download and free < need:
            errors.append(
                f"{runtime} cache {cache} has {free / 1024**3:.2f} GiB free; "
                f"need {need / 1024**3:.2f} GiB for CLIP peak+temp+headroom "
                f"(not the Mac 25 GiB floor)"
            )
        elif not allow_download and cached and free < ARTIFACTS_HEADROOM_BYTES:
            errors.append(
                f"{runtime} cache {cache} has {free / 1024**3:.2f} GiB free; "
                f"need {ARTIFACTS_HEADROOM_BYTES / 1024**3:.2f} GiB artifact headroom"
            )

    report = {
        "runtime": runtime,
        "cache_dir": str(cache),
        "free_bytes": free,
        "free_gib": round(free / 1024**3, 3),
        "mac_min_free_gib": mac_min_free_gib if runtime == "macos" else None,
        "ok": not errors,
        "errors": errors,
        **budget,
    }
    if errors:
        raise PreflightError("Disk preflight failed: " + "; ".join(errors))
    return report


def collect_environment() -> dict:
    env: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime": detect_runtime(),
        "torch": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "is_t4": False,
        "transformers": None,
        "pillow": None,
        "clip_api": {"get_image_features": False, "get_text_features": False},
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            env["gpu_name"] = name
            env["is_t4"] = "T4" in name.upper()
    except ImportError:
        pass
    try:
        import transformers

        env["transformers"] = transformers.__version__
        from transformers import CLIPModel

        env["clip_api"]["get_image_features"] = hasattr(CLIPModel, "get_image_features")
        env["clip_api"]["get_text_features"] = hasattr(CLIPModel, "get_text_features")
    except ImportError:
        pass
    try:
        import PIL

        env["pillow"] = getattr(PIL, "__version__", None)
    except ImportError:
        pass
    return env


def check_environment(
    *,
    runtime: str | None = None,
    require_t4_on_kaggle: bool = True,
) -> dict:
    runtime = runtime or detect_runtime()
    env = collect_environment()
    errors: list[str] = []
    if env["transformers"] is None:
        errors.append("transformers is not installed")
    else:
        parsed = _parse_version(str(env["transformers"]))
        if parsed < TRANSFORMERS_MIN or parsed >= TRANSFORMERS_MAX_EXCLUSIVE:
            errors.append(
                f"transformers {env['transformers']} is outside "
                f"{TRANSFORMERS_MIN} (inclusive) .. {TRANSFORMERS_MAX_EXCLUSIVE} (exclusive); "
                "CLIP get_image_features / get_text_features must be present"
            )
        if not env["clip_api"]["get_image_features"] or not env["clip_api"]["get_text_features"]:
            errors.append("CLIPModel is missing get_image_features or get_text_features")
    if env["pillow"] is None:
        errors.append("Pillow is not installed")
    if env["torch"] is None:
        errors.append("torch is not installed")

    if runtime == "kaggle":
        if not env["cuda_available"]:
            errors.append("Kaggle session has no CUDA; set Accelerator to GPU T4")
        elif require_t4_on_kaggle and not env["is_t4"]:
            errors.append(
                f"Kaggle GPU is {env['gpu_name']!r}, not a T4. "
                "Switch Settings → Accelerator → GPU T4. P100 is not supported."
            )
    env["ok"] = not errors
    env["errors"] = errors
    if errors:
        raise PreflightError("Environment preflight failed: " + "; ".join(errors))
    return env


def smoke_inference(clip: object, image_path: Path, caption: str) -> dict:
    """Tiny real-image encode + L2-sum before the full val loop."""
    import numpy as np
    from styleweave.retrieval.compose import compose_queries
    from styleweave.retrieval.metrics import ranks_from_scores, recall_at_k

    if not image_path.is_file():
        raise PreflightError(f"smoke test image missing: {image_path}")
    images = clip.encode_images([image_path], batch_size=1)
    texts = clip.encode_texts([caption], batch_size=1, max_length=77)
    if images.shape != (1, 512) or texts.shape != (1, 512):
        raise PreflightError(f"unexpected CLIP dim image={images.shape} text={texts.shape}")
    if not np.isfinite(images).all() or not np.isfinite(texts).all():
        raise PreflightError("smoke test produced non-finite CLIP features")
    query = compose_queries(images, texts, mode="l2_sum")
    ranks = ranks_from_scores(query, images, np.array([0]))
    hit = recall_at_k(ranks, 1)
    if hit != 1.0:
        raise PreflightError("smoke ranking failed to retrieve the encoded image at rank 1")
    return {
        "ok": True,
        "image": str(image_path),
        "caption": caption,
        "image_shape": list(images.shape),
        "text_shape": list(texts.shape),
        "recall_at_1": hit,
    }
