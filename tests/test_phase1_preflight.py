"""Preflight, coverage, and stale-metrics tests. No CLIP download."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from styleweave.artifacts import invalidate_metrics
from styleweave.config import B0Config
from styleweave.data.coverage import check_category_coverage
from styleweave.eval.b0 import CoverageError, run_b0
from styleweave.preflight import (
    PreflightError,
    check_disk_budget,
    check_environment,
    collect_environment,
    smoke_inference,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "fashioniq_tiny"


def _touch_images(images_dir: Path, ids: list[str]) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for image_id in ids:
        (images_dir / f"{image_id}.jpg").write_bytes(b"not-a-real-jpeg")


class _Disk:
    def __init__(self, free: int) -> None:
        self.free = free
        self.used = 0
        self.total = free


class DiskPolicyTests(unittest.TestCase):
    def test_macos_keeps_25_gib_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("styleweave.preflight.shutil.disk_usage", return_value=_Disk(10 * 1024**3)):
                with self.assertRaises(PreflightError) as ctx:
                    check_disk_budget(
                        runtime="macos",
                        cache_dir=tmp,
                        allow_download=True,
                        mac_min_free_gib=25.0,
                    )
        self.assertIn("25", str(ctx.exception))

    def test_kaggle_uses_clip_footprint_not_mac_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("styleweave.preflight.shutil.disk_usage", return_value=_Disk(3 * 1024**3)):
                report = check_disk_budget(
                    runtime="kaggle",
                    cache_dir=tmp,
                    allow_download=True,
                )
        self.assertTrue(report["ok"])
        self.assertIsNone(report["mac_min_free_gib"])
        self.assertGreater(report["required_free_bytes"], 605_247_071)
        self.assertLess(report["required_free_bytes"], 25 * 1024**3)

    def test_kaggle_fails_when_cache_volume_is_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("styleweave.preflight.shutil.disk_usage", return_value=_Disk(100 * 1024**2)):
                with self.assertRaises(PreflightError) as ctx:
                    check_disk_budget(
                        runtime="kaggle",
                        cache_dir=tmp,
                        allow_download=True,
                    )
        self.assertIn("not the Mac 25 GiB floor", str(ctx.exception))


class CoverageHardeningTests(unittest.TestCase):
    def test_missing_reference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            _touch_images(images, ["IMG_B", "IMG_C", "IMG_D", "IMG_E"])
            report, _, _ = check_category_coverage(
                captions_dir=FIXTURE / "captions",
                splits_dir=FIXTURE / "image_splits",
                images_dir=images,
                category="dress",
                split="val",
                caption_mode="clip4cir_concat",
                allow_missing_gallery_frac=0.0,
                fail_on_missing_target=True,
                fail_on_missing_reference=True,
                require_complete_gallery=True,
            )
            self.assertFalse(report.ok)
            self.assertIn("IMG_A", report.missing_reference_ids)

    def test_target_not_in_gallery_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captions = root / "captions"
            splits = root / "image_splits"
            images = root / "images"
            captions.mkdir()
            splits.mkdir()
            (captions / "cap.dress.val.json").write_text(
                json.dumps(
                    [
                        {
                            "target": "IMG_Z",
                            "candidate": "IMG_A",
                            "captions": ["is darker", "more casual"],
                        }
                    ]
                )
            )
            (splits / "split.dress.val.json").write_text(json.dumps(["IMG_A", "IMG_B"]))
            _touch_images(images, ["IMG_A", "IMG_B", "IMG_Z"])
            report, _, _ = check_category_coverage(
                captions_dir=captions,
                splits_dir=splits,
                images_dir=images,
                category="dress",
                split="val",
                caption_mode="clip4cir_concat",
                allow_missing_gallery_frac=0.0,
                fail_on_missing_target=True,
                fail_on_missing_reference=True,
                require_complete_gallery=True,
            )
            self.assertFalse(report.ok)
            self.assertIn("IMG_Z", report.targets_not_in_gallery)

    def test_incomplete_gallery_fails_without_justification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            _touch_images(images, ["IMG_A", "IMG_B", "IMG_C", "IMG_D"])
            report, _, _ = check_category_coverage(
                captions_dir=FIXTURE / "captions",
                splits_dir=FIXTURE / "image_splits",
                images_dir=images,
                category="dress",
                split="val",
                caption_mode="clip4cir_concat",
                allow_missing_gallery_frac=0.5,
                fail_on_missing_target=True,
                fail_on_missing_reference=True,
                require_complete_gallery=True,
            )
            self.assertFalse(report.ok)
            self.assertTrue(any("complete gallery" in err for err in report.errors))

    def test_incomplete_gallery_allowed_only_with_justification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            _touch_images(images, ["IMG_A", "IMG_B", "IMG_C", "IMG_D"])
            report, _, _ = check_category_coverage(
                captions_dir=FIXTURE / "captions",
                splits_dir=FIXTURE / "image_splits",
                images_dir=images,
                category="dress",
                split="val",
                caption_mode="clip4cir_concat",
                allow_missing_gallery_frac=0.5,
                fail_on_missing_target=True,
                fail_on_missing_reference=True,
                require_complete_gallery=False,
                incomplete_gallery_justification="unit-test documented research exception",
            )
            self.assertTrue(report.ok)
            self.assertEqual(report.n_missing_gallery, 1)


class StaleMetricsTests(unittest.TestCase):
    def test_failed_coverage_quarantines_previous_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "metrics.json").write_text(
                json.dumps({"produced_by": "styleweave.eval.b0", "stale": True})
            )
            _touch_images(images, ["IMG_A", "IMG_C", "IMG_E"])
            cfg = B0Config(
                captions_dir=str(FIXTURE / "captions"),
                image_splits_dir=str(FIXTURE / "image_splits"),
                images_dir=str(images),
                artifacts_dir=str(artifacts),
                categories=("dress",),
            )
            with self.assertRaises(CoverageError):
                run_b0(cfg, dry_run=True, images_source="fixture")
            self.assertFalse((artifacts / "metrics.json").is_file())
            invalidated = list(artifacts.glob("metrics.json.invalidated-*"))
            self.assertEqual(len(invalidated), 1)
            status = json.loads((artifacts / "run_status.json").read_text())
            self.assertEqual(status["status"], "invalidated")

    def test_full_run_start_quarantines_metrics_before_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "metrics.json").write_text(
                json.dumps({"produced_by": "styleweave.eval.b0", "stale": True})
            )
            _touch_images(images, ["IMG_A", "IMG_C", "IMG_E"])
            cfg = B0Config(
                captions_dir=str(FIXTURE / "captions"),
                image_splits_dir=str(FIXTURE / "image_splits"),
                images_dir=str(images),
                artifacts_dir=str(artifacts),
                categories=("dress",),
            )
            with self.assertRaises(CoverageError):
                run_b0(cfg, dry_run=False, images_source="fixture")
            self.assertFalse((artifacts / "metrics.json").is_file())
            self.assertGreaterEqual(len(list(artifacts.glob("metrics.json.invalidated-*"))), 1)

    def test_invalidate_helper_moves_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            (artifacts / "metrics.json").write_text("{}")
            moved = invalidate_metrics(artifacts, "unit-test")
            self.assertIsNotNone(moved)
            self.assertFalse((artifacts / "metrics.json").exists())
            assert moved is not None
            self.assertTrue(moved.exists())


class SmokeTests(unittest.TestCase):
    def test_smoke_ranking_with_fake_clip(self) -> None:
        class FakeCLIP:
            def encode_images(self, paths, batch_size=32):
                return np.ones((len(paths), 512), dtype=np.float32)

            def encode_texts(self, texts, batch_size=32, max_length=77):
                return np.ones((len(texts), 512), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "real.jpg"
            image.write_bytes(b"not-decoded-by-fake-clip")
            result = smoke_inference(FakeCLIP(), image, "is darker and more casual")
        self.assertTrue(result["ok"])
        self.assertEqual(result["recall_at_1"], 1.0)

    def test_collect_environment_does_not_download(self) -> None:
        env = collect_environment()
        self.assertIn("python", env)
        self.assertIn("clip_api", env)
        self.assertIn("transformers", env)
        self.assertIn("pillow", env)


class EnvironmentGateTests(unittest.TestCase):
    def _env(self, **overrides: object) -> dict:
        base = {
            "python": "3.10.12",
            "platform": "Linux",
            "runtime": "kaggle",
            "torch": "2.4.0+cu121",
            "cuda_available": True,
            "cuda_version": "12.1",
            "gpu_name": "Tesla T4",
            "is_t4": True,
            "transformers": "4.44.2",
            "pillow": "10.4.0",
            "clip_api": {"get_image_features": True, "get_text_features": True},
        }
        base.update(overrides)
        return base

    def test_kaggle_rejects_non_t4(self) -> None:
        fake = self._env(gpu_name="Tesla P100-PCIE-16GB", is_t4=False)
        with patch("styleweave.preflight.collect_environment", return_value=fake):
            with self.assertRaises(PreflightError) as ctx:
                check_environment(runtime="kaggle", require_t4_on_kaggle=True)
        self.assertIn("T4", str(ctx.exception))

    def test_transformers_pin_rejects_too_new(self) -> None:
        fake = self._env(transformers="5.3.0")
        with patch("styleweave.preflight.collect_environment", return_value=fake):
            with self.assertRaises(PreflightError) as ctx:
                check_environment(runtime="other", require_t4_on_kaggle=False)
        self.assertIn("transformers", str(ctx.exception))

    def test_kaggle_t4_and_clip_api_pass(self) -> None:
        fake = self._env()
        with patch("styleweave.preflight.collect_environment", return_value=fake):
            report = check_environment(runtime="kaggle", require_t4_on_kaggle=True)
        self.assertTrue(report["ok"])


class NotebookContractTests(unittest.TestCase):
    def test_notebook_fails_clearly_on_preflight(self) -> None:
        nb = json.loads((ROOT / "notebooks" / "kaggle_phase1_b0.ipynb").read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
        self.assertIn("check_environment(require_t4_on_kaggle=True)", source)
        self.assertIn("ENVIRONMENT FAILED", source)
        self.assertIn("DISK FAILED", source)
        self.assertIn("COVERAGE FAILED", source)
        self.assertIn("transformers>=4.41,<5.3", source)
        self.assertIn("GPU T4", source)
        self.assertIn("No Qwen. No SigLIP2", source)
        self.assertNotIn("google/siglip", source.lower())
        self.assertNotIn("Qwen2-VL", source)


if __name__ == "__main__":
    unittest.main()
