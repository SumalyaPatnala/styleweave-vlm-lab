"""Publish eval artifacts to Hugging Face Hub. Images are never uploaded."""

from __future__ import annotations

import os
from pathlib import Path

from styleweave.config import HubConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
BLOCKED_DIR_NAMES = {"images", "raw", "fashion-iq", "fashioniq"}


class PublishDisabled(RuntimeError):
    pass


class PublishRefused(RuntimeError):
    pass


def publishing_enabled(hub: HubConfig, *, flag: bool) -> bool:
    env_name = hub.require_env
    env_value = os.environ.get(env_name, "0").strip().lower()
    env_on = env_value in {"1", "true", "yes"}
    return bool(flag and hub.enabled and env_on)


def _assert_no_images(paths: list[Path]) -> None:
    for path in paths:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            raise PublishRefused(f"Refusing to upload image file: {path}")
        parts = {part.lower() for part in path.parts}
        if parts & BLOCKED_DIR_NAMES:
            raise PublishRefused(
                f"Refusing to upload path under a dataset/image directory: {path}"
            )


def publish_eval_artifacts(
    *,
    hub: HubConfig,
    artifacts_dir: Path,
    extra_files: list[Path] | None = None,
    flag: bool = False,
) -> dict:
    if not publishing_enabled(hub, flag=flag):
        raise PublishDisabled(
            "Hugging Face publishing is disabled. It requires hub.enabled=true "
            f"in config, --publish-hub, and {hub.require_env}=1. Default is off."
        )
    if hub.allow_images:
        raise PublishRefused("allow_images is hard-denied for FashionIQ")
    if not hub.repo_id:
        raise PublishRefused("hub.repo_id is empty")

    files = []
    for name in ("metrics.json", "report.md", "coverage.json"):
        path = artifacts_dir / name
        if path.is_file():
            files.append(path)
    if extra_files:
        files.extend(extra_files)
    if not files:
        raise PublishRefused("no eval artifacts to publish")
    _assert_no_images(files)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise PublishRefused("huggingface_hub is not installed") from exc

    api = HfApi()
    api.create_repo(hub.repo_id, private=hub.private, exist_ok=True, repo_type="model")
    uploaded = []
    for path in files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=hub.repo_id,
            repo_type="model",
        )
        uploaded.append(path.name)
    return {"repo_id": hub.repo_id, "uploaded": uploaded, "images_uploaded": False}
