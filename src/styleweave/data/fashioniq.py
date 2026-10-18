"""Official FashionIQ captions, splits, and image path resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from styleweave.retrieval.compose import combine_relative_captions

IMAGE_SUFFIXES = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


@dataclass(frozen=True)
class FashionIQQuery:
    category: str
    reference_id: str
    target_id: str
    captions: tuple[str, ...]
    text: str


def caption_path(captions_dir: Path, category: str, split: str) -> Path:
    return captions_dir / f"cap.{category}.{split}.json"


def split_path(splits_dir: Path, category: str, split: str) -> Path:
    return splits_dir / f"split.{category}.{split}.json"


def load_json(path: Path) -> object:
    with path.open() as handle:
        return json.load(handle)


def load_split_ids(splits_dir: Path, category: str, split: str) -> list[str]:
    payload = load_json(split_path(splits_dir, category, split))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"{split_path(splits_dir, category, split)} must be a list of IDs")
    return payload


def load_queries(
    captions_dir: Path,
    category: str,
    split: str,
    caption_mode: str = "clip4cir_concat",
) -> list[FashionIQQuery]:
    payload = load_json(caption_path(captions_dir, category, split))
    if not isinstance(payload, list):
        raise ValueError(f"{caption_path(captions_dir, category, split)} must be a list")
    queries: list[FashionIQQuery] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("caption rows must be objects")
        try:
            reference_id = str(row["candidate"])
            target_id = str(row["target"])
            captions = tuple(str(item) for item in row["captions"])
        except KeyError as exc:
            raise ValueError(f"caption row missing {exc}") from exc
        queries.append(
            FashionIQQuery(
                category=category,
                reference_id=reference_id,
                target_id=target_id,
                captions=captions,
                text=combine_relative_captions(list(captions), mode=caption_mode),
            )
        )
    return queries


def iter_image_candidates(images_dir: Path, image_id: str) -> list[Path]:
    names = [f"{image_id}{suffix}" for suffix in IMAGE_SUFFIXES]
    candidates: list[Path] = []
    for name in names:
        candidates.append(images_dir / name)
        for sub in ("dress", "shirt", "toptee", "top-tee", "images"):
            candidates.append(images_dir / sub / name)
    return candidates


def resolve_image(images_dir: Path, image_id: str) -> Path | None:
    for path in iter_image_candidates(images_dir, image_id):
        if path.is_file():
            return path
    return None


def discover_fashioniq_layout(root: Path) -> dict[str, Path] | None:
    """Find captions/, image_splits/, images/ under common Kaggle/local layouts."""
    candidates = [
        root,
        root / "fashionIQ_dataset",
        root / "fashion-iq",
        root / "FashionIQ",
        root / "data" / "raw" / "fashion-iq",
    ]
    for base in candidates:
        captions = base / "captions"
        splits = base / "image_splits"
        images = base / "images"
        if captions.is_dir() and splits.is_dir():
            return {
                "root": base,
                "captions_dir": captions,
                "image_splits_dir": splits,
                "images_dir": images if images.is_dir() else base,
            }
    return None
