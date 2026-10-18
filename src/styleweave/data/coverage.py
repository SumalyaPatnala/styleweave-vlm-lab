"""Validate FashionIQ provenance and image coverage before any encoding."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from styleweave.data.fashioniq import (
    FashionIQQuery,
    load_queries,
    load_split_ids,
    resolve_image,
)

LICENSE_NOTE = (
    "FashionIQ official README names CDLA without Permissive vs Sharing. "
    "GitHub issue #22 is unanswered. Academic research use only; do not "
    "redistribute images."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CoverageReport:
    category: str
    split: str
    n_gallery: int
    n_queries: int
    n_missing_gallery: int
    n_missing_targets: int
    n_missing_references: int
    missing_gallery_frac: float
    missing_target_ids: list[str] = field(default_factory=list)
    missing_gallery_ids: list[str] = field(default_factory=list)
    caption_sha256: str = ""
    split_sha256: str = ""
    ok: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "split": self.split,
            "n_gallery": self.n_gallery,
            "n_queries": self.n_queries,
            "n_missing_gallery": self.n_missing_gallery,
            "n_missing_targets": self.n_missing_targets,
            "n_missing_references": self.n_missing_references,
            "missing_gallery_frac": self.missing_gallery_frac,
            "missing_target_ids": self.missing_target_ids[:50],
            "missing_gallery_ids": self.missing_gallery_ids[:50],
            "caption_sha256": self.caption_sha256,
            "split_sha256": self.split_sha256,
            "ok": self.ok,
            "errors": self.errors,
        }


def check_category_coverage(
    *,
    captions_dir: Path,
    splits_dir: Path,
    images_dir: Path,
    category: str,
    split: str,
    caption_mode: str,
    allow_missing_gallery_frac: float,
    fail_on_missing_target: bool,
) -> tuple[CoverageReport, list[str], list[FashionIQQuery]]:
    from styleweave.data.fashioniq import caption_path, split_path

    cap_file = caption_path(captions_dir, category, split)
    split_file = split_path(splits_dir, category, split)
    if not cap_file.is_file():
        raise FileNotFoundError(cap_file)
    if not split_file.is_file():
        raise FileNotFoundError(split_file)

    gallery_ids = load_split_ids(splits_dir, category, split)
    queries = load_queries(captions_dir, category, split, caption_mode=caption_mode)
    gallery_set = set(gallery_ids)

    missing_gallery = [image_id for image_id in gallery_ids if resolve_image(images_dir, image_id) is None]
    missing_targets = sorted(
        {
            query.target_id
            for query in queries
            if resolve_image(images_dir, query.target_id) is None
        }
    )
    missing_refs = sum(
        1 for query in queries if resolve_image(images_dir, query.reference_id) is None
    )
    frac = (len(missing_gallery) / len(gallery_ids)) if gallery_ids else 1.0

    report = CoverageReport(
        category=category,
        split=split,
        n_gallery=len(gallery_ids),
        n_queries=len(queries),
        n_missing_gallery=len(missing_gallery),
        n_missing_targets=len(missing_targets),
        n_missing_references=missing_refs,
        missing_gallery_frac=frac,
        missing_target_ids=missing_targets,
        missing_gallery_ids=missing_gallery,
        caption_sha256=sha256_file(cap_file),
        split_sha256=sha256_file(split_file),
        ok=True,
    )
    if fail_on_missing_target and missing_targets:
        report.ok = False
        report.errors.append(
            f"{category}: {len(missing_targets)} missing val targets (fail_on_missing_target)"
        )
    if frac > allow_missing_gallery_frac:
        report.ok = False
        report.errors.append(
            f"{category}: missing gallery fraction {frac:.4f} > {allow_missing_gallery_frac}"
        )
    if not gallery_ids:
        report.ok = False
        report.errors.append(f"{category}: empty gallery split")
    return report, gallery_ids, queries


def split_disjointness(
    splits_dir: Path, category: str
) -> dict[str, object]:
    train = set(load_split_ids(splits_dir, category, "train")) if (splits_dir / f"split.{category}.train.json").is_file() else set()
    val = set(load_split_ids(splits_dir, category, "val"))
    overlap = sorted(train & val)
    return {
        "category": category,
        "train_n": len(train),
        "val_n": len(val),
        "train_val_overlap_n": len(overlap),
        "train_val_overlap_sample": overlap[:20],
    }


def provenance_payload(
    *,
    images_source: str,
    captions_dir: Path,
    splits_dir: Path,
    images_dir: Path,
    reports: list[CoverageReport],
) -> dict:
    return {
        "dataset": "fashion-iq",
        "license_note": LICENSE_NOTE,
        "captions_dir": str(captions_dir),
        "image_splits_dir": str(splits_dir),
        "images_dir": str(images_dir),
        "images_source": images_source,
        "do_not_redistribute_images": True,
        "coverage": [item.to_dict() for item in reports],
    }
