"""Command-line entry for Phase 1 B0. No Qwen. No paid GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from styleweave.config import B0Config, ROOT, load_b0_config
from styleweave.data.fashioniq import discover_fashioniq_layout
from styleweave.eval.b0 import CoverageError, run_b0
from styleweave.hub.publish import PublishDisabled, PublishRefused, publish_eval_artifacts


def _apply_layout(cfg: B0Config, data_root: Path | None) -> B0Config:
    if data_root is None:
        return cfg
    layout = discover_fashioniq_layout(data_root)
    if layout is None:
        raise FileNotFoundError(
            f"No FashionIQ captions/ + image_splits/ under {data_root}"
        )
    cfg.captions_dir = str(layout["captions_dir"])
    cfg.image_splits_dir = str(layout["image_splits_dir"])
    cfg.images_dir = str(layout["images_dir"])
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StyleWeave Phase 1 B0 runner")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None, help="Auto-discover FashionIQ layout")
    parser.add_argument("--captions-dir", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--images-source", type=str, default="unspecified")
    parser.add_argument("--device", type=str, default=None, help="Force cuda, mps, or cpu")
    parser.add_argument("--allow-download", action="store_true", help="Allow CLIP weight download after disk check")
    parser.add_argument("--dry-run", action="store_true", help="Coverage + provenance only; no CLIP")
    parser.add_argument("--publish-hub", action="store_true", help="Opt-in HF upload; still needs env flag")
    parser.add_argument("--hub-repo", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_b0_config(args.config, root=ROOT)
    if args.data_root:
        cfg = _apply_layout(cfg, args.data_root)
    if args.captions_dir:
        cfg.captions_dir = str(args.captions_dir)
    if args.splits_dir:
        cfg.image_splits_dir = str(args.splits_dir)
    if args.images_dir:
        cfg.images_dir = str(args.images_dir)
    if args.artifacts_dir:
        cfg.artifacts_dir = str(args.artifacts_dir)
    if args.device:
        cfg.device_preference = [args.device, "cpu"]
    if args.allow_download:
        cfg.allow_download = True
    if args.hub_repo:
        cfg.hub.repo_id = args.hub_repo

    try:
        result = run_b0(
            cfg,
            dry_run=args.dry_run,
            images_source=args.images_source,
            cache_dir=args.cache_dir,
        )
    except CoverageError as exc:
        print(f"COVERAGE FAILED: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"RUN FAILED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({k: result[k] for k in result if k != "coverage"}, indent=2, default=str)[:4000])
    if args.dry_run:
        print("Dry-run complete. metrics.json was not written.")
        return 0

    if args.publish_hub:
        try:
            published = publish_eval_artifacts(
                hub=cfg.hub,
                artifacts_dir=Path(cfg.artifacts_dir),
                flag=True,
            )
            print("Hub upload:", published)
        except (PublishDisabled, PublishRefused) as exc:
            print(f"Hub publish skipped/refused: {exc}", file=sys.stderr)
            return 3
    else:
        print("Hugging Face publishing left disabled (default).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
