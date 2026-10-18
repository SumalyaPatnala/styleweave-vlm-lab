"""Phase 1 B0: frozen CLIP L2-sum on official FashionIQ val."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from styleweave import __version__
from styleweave.artifacts import invalidate_metrics, mark_run
from styleweave.config import B0Config, ROOT
from styleweave.data.coverage import (
    LICENSE_NOTE,
    check_category_coverage,
    provenance_payload,
    split_disjointness,
)
from styleweave.data.fashioniq import resolve_image
from styleweave.device import device_details, select_device
from styleweave.hardware import git_commit, hardware_log
from styleweave.preflight import (
    PreflightError,
    check_disk_budget,
    check_environment,
    collect_environment,
    default_cache_dir,
    detect_runtime,
    smoke_inference,
)
from styleweave.retrieval.compose import compose_queries
from styleweave.retrieval.metrics import ranks_from_scores, recall_at_k
from styleweave.seeds import set_seed

PRODUCED_BY = "styleweave.eval.b0"


class CoverageError(RuntimeError):
    pass


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_report(path: Path, payload: dict) -> None:
    metrics = payload["metrics"]
    mean = metrics["mean"]
    lines = [
        "# Phase 1 B0 report",
        "",
        f"Produced by `{payload['produced_by']}` version {payload['code_version']}.",
        "These numbers are StyleWeave measurements, not CLIP4Cir Table 1.",
        "",
        f"- Model: `{payload['model']['name']}` revision `{payload['model']['revision_resolved']}`",
        f"- Query: `{payload['query_mode']}` (Phase 0 L2-sum)",
        f"- Split: FashionIQ `{payload['data']['split']}`",
        f"- Device: `{payload['hardware'].get('device')}`",
        f"- Wall time: {payload['wall_time_sec']:.1f}s",
        "",
        "| Category | R@10 | R@50 | Queries |",
        "| --- | --- | --- | --- |",
    ]
    for category in payload["data"]["categories"]:
        row = metrics[category]
        lines.append(
            f"| {category} | {row['recall@10'] * 100:.2f} | {row['recall@50'] * 100:.2f} | {row['n_queries']} |"
        )
    lines.extend(
        [
            f"| mean | {mean['recall@10'] * 100:.2f} | {mean['recall@50'] * 100:.2f} | — |",
            "",
            "Literature orientation (not our result): "
            f"{payload['literature_orientation']['source']} "
            f"mean R@10/R@50 {payload['literature_orientation']['mean_recall_at_10']:.2f} / "
            f"{payload['literature_orientation']['mean_recall_at_50']:.2f}.",
            "",
            LICENSE_NOTE,
            "",
        ]
    )
    path.write_text("\n".join(lines))


def validate_coverage(cfg: B0Config, images_source: str = "unspecified") -> dict:
    captions_dir = Path(cfg.captions_dir)
    splits_dir = Path(cfg.image_splits_dir)
    images_dir = Path(cfg.images_dir)
    reports = []
    errors: list[str] = []
    disjoint = []
    for category in cfg.categories:
        report, _, _ = check_category_coverage(
            captions_dir=captions_dir,
            splits_dir=splits_dir,
            images_dir=images_dir,
            category=category,
            split=cfg.report_split,
            caption_mode=cfg.caption_mode,
            allow_missing_gallery_frac=cfg.allow_missing_gallery_frac,
            fail_on_missing_target=cfg.fail_on_missing_target,
            fail_on_missing_reference=cfg.fail_on_missing_reference,
            require_complete_gallery=cfg.require_complete_gallery,
            incomplete_gallery_justification=cfg.incomplete_gallery_justification,
        )
        reports.append(report)
        errors.extend(report.errors)
        disjoint.append(split_disjointness(splits_dir, category))
    payload = provenance_payload(
        images_source=images_source,
        captions_dir=captions_dir,
        splits_dir=splits_dir,
        images_dir=images_dir,
        reports=reports,
    )
    payload["split_disjointness"] = disjoint
    payload["ok"] = not errors
    payload["errors"] = errors
    return payload


def run_b0(
    cfg: B0Config,
    *,
    dry_run: bool = False,
    images_source: str = "unspecified",
    cache_dir: str | None = None,
    skip_env_check: bool = False,
) -> dict:
    started = time.time()
    set_seed(cfg.seed)
    artifacts = Path(cfg.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    runtime = detect_runtime()
    resolved_cache = str(default_cache_dir(runtime, cache_dir))

    if not dry_run:
        invalidate_metrics(artifacts, "full B0 run started; previous metrics are not this run")
        mark_run(artifacts, "in_progress", extra={"runtime": runtime})

    coverage = validate_coverage(cfg, images_source=images_source)
    _write_json(artifacts / "coverage.json", coverage)
    if not coverage["ok"]:
        invalidate_metrics(artifacts, "coverage preflight failed")
        raise CoverageError(
            "FashionIQ coverage failed; metrics.json is not an official result. "
            + "; ".join(coverage["errors"])
        )

    if dry_run:
        mark_run(artifacts, "dry_run_ok", extra={"metrics_written": False})
        return {"dry_run": True, "coverage": coverage, "metrics_written": False}

    os.environ["HF_HOME"] = resolved_cache
    os.environ["TRANSFORMERS_CACHE"] = resolved_cache

    try:
        env = (
            collect_environment()
            if skip_env_check
            else check_environment(runtime=runtime, require_t4_on_kaggle=True)
        )
        disk = check_disk_budget(
            runtime=runtime,
            cache_dir=resolved_cache,
            allow_download=cfg.allow_download,
            model_name=cfg.model_name,
            mac_min_free_gib=cfg.min_free_gib,
        )

        import numpy as np

        device_name = select_device(cfg.device_preference)
        from styleweave.models.clip_frozen import FrozenCLIP

        clip = FrozenCLIP(
            cfg.model_name,
            cfg.model_revision,
            device_name,
            allow_download=cfg.allow_download,
            min_free_gib=cfg.min_free_gib,
            cache_dir=resolved_cache,
            frozen=cfg.frozen,
            skip_disk_check=True,
        )
        if clip.trainable_module_names():
            raise RuntimeError(f"CLIP is not frozen: {clip.trainable_module_names()}")

        images_dir = Path(cfg.images_dir)
        captions_dir = Path(cfg.captions_dir)
        splits_dir = Path(cfg.image_splits_dir)
        per_category: dict[str, dict] = {}
        smoke = None

        for category in cfg.categories:
            _report, gallery_ids, queries = check_category_coverage(
                captions_dir=captions_dir,
                splits_dir=splits_dir,
                images_dir=images_dir,
                category=category,
                split=cfg.report_split,
                caption_mode=cfg.caption_mode,
                allow_missing_gallery_frac=cfg.allow_missing_gallery_frac,
                fail_on_missing_target=cfg.fail_on_missing_target,
                fail_on_missing_reference=cfg.fail_on_missing_reference,
                require_complete_gallery=cfg.require_complete_gallery,
                incomplete_gallery_justification=cfg.incomplete_gallery_justification,
            )
            if not _report.ok:
                raise CoverageError("; ".join(_report.errors))
            gallery_paths = [resolve_image(images_dir, image_id) for image_id in gallery_ids]
            if any(path is None for path in gallery_paths):
                raise CoverageError(
                    f"{category}: gallery path vanished after coverage check"
                )
            if smoke is None:
                first = queries[0]
                ref_path = resolve_image(images_dir, first.reference_id)
                assert ref_path is not None
                smoke = smoke_inference(clip, ref_path, first.text)
            id_to_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
            gallery_features = clip.encode_images(
                [path for path in gallery_paths if path is not None],
                batch_size=cfg.batch_size,
            )
            query_image_ids = [query.reference_id for query in queries]
            unique_refs = list(dict.fromkeys(query_image_ids))
            ref_paths = []
            for image_id in unique_refs:
                path = resolve_image(images_dir, image_id)
                if path is None:
                    raise CoverageError(f"{category}: missing reference {image_id}")
                ref_paths.append(path)
            ref_features = clip.encode_images(ref_paths, batch_size=cfg.batch_size)
            ref_map = {
                image_id: ref_features[index]
                for index, image_id in enumerate(unique_refs)
            }
            image_features = np.stack(
                [ref_map[query.reference_id] for query in queries]
            )
            text_features = clip.encode_texts(
                [query.text for query in queries],
                batch_size=cfg.batch_size,
                max_length=cfg.max_text_length,
            )
            query_features = compose_queries(
                image_features, text_features, mode=cfg.query_mode
            )
            target_indices = np.array(
                [id_to_index[query.target_id] for query in queries]
            )
            ranks = ranks_from_scores(query_features, gallery_features, target_indices)
            per_category[category] = {
                "recall@10": recall_at_k(ranks, 10),
                "recall@50": recall_at_k(ranks, 50),
                "n_queries": int(len(queries)),
                "n_gallery": int(len(gallery_ids)),
                "mean_rank": float(ranks.mean()),
            }

        mean_r10 = float(
            np.mean([per_category[cat]["recall@10"] for cat in cfg.categories])
        )
        mean_r50 = float(
            np.mean([per_category[cat]["recall@50"] for cat in cfg.categories])
        )
        metrics = {
            **per_category,
            "mean": {"recall@10": mean_r10, "recall@50": mean_r50},
        }
        payload = {
            "schema_version": "1.0",
            "produced_by": PRODUCED_BY,
            "code_version": __version__,
            "experiment_id": cfg.experiment_id,
            "query_mode": cfg.query_mode,
            "caption_mode": cfg.caption_mode,
            "seed": cfg.seed,
            "git_commit": git_commit(ROOT),
            "model": clip.card(),
            "data": {
                "name": "fashion-iq",
                "split": cfg.report_split,
                "categories": list(cfg.categories),
                "provenance": coverage,
            },
            "metrics": metrics,
            "preflight": {"environment": env, "disk": disk, "smoke": smoke},
            "hardware": hardware_log(device_name, extra=device_details(device_name)),
            "wall_time_sec": time.time() - started,
            "literature_orientation": {
                "source": cfg.literature_source,
                "mean_recall_at_10": cfg.literature_r10,
                "mean_recall_at_50": cfg.literature_r50,
                "note": "Published CLIP4Cir numbers for orientation only. Not StyleWeave results.",
            },
            "hub_published": False,
            "qwen_used": False,
            "siglip2_used": False,
        }
        _write_json(artifacts / "metrics.json", payload)
        _write_report(artifacts / "report.md", payload)
        mark_run(artifacts, "complete", extra={"metrics_written": True})
        return payload
    except (CoverageError, PreflightError) as exc:
        invalidate_metrics(artifacts, str(exc))
        raise
    except Exception as exc:
        invalidate_metrics(artifacts, f"{type(exc).__name__}: {exc}")
        mark_run(artifacts, "failed", extra={"error": str(exc)})
        raise
