"""Phase 1 B0 unit tests. No CLIP download. No FashionIQ images from the web."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from styleweave.config import B0Config, HubConfig
from styleweave.data.coverage import check_category_coverage, split_disjointness
from styleweave.data.fashioniq import discover_fashioniq_layout, load_queries
from styleweave.eval.b0 import CoverageError, run_b0
from styleweave.hub.publish import PublishDisabled, PublishRefused, publishing_enabled, publish_eval_artifacts
from styleweave.retrieval.compose import combine_relative_captions, compose_queries, l2_normalize
from styleweave.retrieval.metrics import ranks_from_scores, recall_at_k

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "fashioniq_tiny"


def _touch_images(images_dir: Path, ids: list[str]) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for image_id in ids:
        (images_dir / f"{image_id}.jpg").write_bytes(b"not-a-real-jpeg")


class ComposeAndRecallTests(unittest.TestCase):
    def test_l2_sum_matches_phase0_formula(self) -> None:
        image = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
        text = np.array([[0.0, 4.0, 0.0]], dtype=np.float32)
        query = compose_queries(image, text, mode="l2_sum")
        expected = l2_normalize(l2_normalize(image) + l2_normalize(text))
        np.testing.assert_allclose(query, expected, atol=1e-6)

    def test_recall_at_k_on_four_item_gallery(self) -> None:
        gallery = np.eye(4, dtype=np.float32)
        queries = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.1, 0.0, 0.0, 0.9],
            ],
            dtype=np.float32,
        )
        targets = np.array([0, 1, 2, 3])
        ranks = ranks_from_scores(queries, gallery, targets)
        self.assertTrue(np.all(ranks == 0))
        self.assertEqual(recall_at_k(ranks, 10), 1.0)
        self.assertEqual(recall_at_k(ranks, 1), 1.0)

        missed = ranks_from_scores(
            np.array([[0.01, 0.1, 0.2, 1.0]], dtype=np.float32),
            gallery,
            np.array([0]),
        )
        self.assertEqual(int(missed[0]), 3)
        self.assertEqual(recall_at_k(missed, 1), 0.0)
        self.assertEqual(recall_at_k(missed, 50), 1.0)

    def test_clip4cir_caption_concat(self) -> None:
        text = combine_relative_captions(
            ["is shiny and silver with shorter sleeves", "fit and flare"]
        )
        self.assertEqual(text, "Is shiny and silver with shorter sleeves and fit and flare")


class CoverageTests(unittest.TestCase):
    def test_coverage_fails_on_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            images = tmp_path / "images"
            _touch_images(images, ["IMG_A", "IMG_C", "IMG_E"])
            report, _, queries = check_category_coverage(
                captions_dir=FIXTURE / "captions",
                splits_dir=FIXTURE / "image_splits",
                images_dir=images,
                category="dress",
                split="val",
                caption_mode="clip4cir_concat",
                allow_missing_gallery_frac=0.01,
                fail_on_missing_target=True,
            )
            self.assertGreaterEqual(len(queries), 1)
            self.assertFalse(report.ok)
            self.assertIn("IMG_B", report.missing_target_ids)
            self.assertIn("IMG_D", report.missing_target_ids)

    def test_coverage_passes_when_all_val_ids_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            _touch_images(images, ["IMG_A", "IMG_B", "IMG_C", "IMG_D", "IMG_E"])
            report, gallery, queries = check_category_coverage(
                captions_dir=FIXTURE / "captions",
                splits_dir=FIXTURE / "image_splits",
                images_dir=images,
                category="dress",
                split="val",
                caption_mode="clip4cir_concat",
                allow_missing_gallery_frac=0.01,
                fail_on_missing_target=True,
            )
            self.assertTrue(report.ok)
            self.assertEqual(len(gallery), 5)
            self.assertEqual(len(queries), 2)
            disjoint = split_disjointness(FIXTURE / "image_splits", "dress")
            self.assertEqual(disjoint["train_val_overlap_n"], 0)

    def test_dry_run_does_not_write_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            _touch_images(images, ["IMG_A", "IMG_B", "IMG_C", "IMG_D", "IMG_E"])
            artifacts = root / "artifacts"
            cfg = B0Config(
                captions_dir=str(FIXTURE / "captions"),
                image_splits_dir=str(FIXTURE / "image_splits"),
                images_dir=str(images),
                artifacts_dir=str(artifacts),
                categories=("dress",),
            )
            result = run_b0(cfg, dry_run=True, images_source="fixture")
            self.assertTrue(result["dry_run"])
            self.assertFalse((artifacts / "metrics.json").exists())
            self.assertTrue((artifacts / "coverage.json").exists())

    def test_missing_target_refuses_metrics_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            _touch_images(images, ["IMG_A", "IMG_C", "IMG_E"])
            cfg = B0Config(
                captions_dir=str(FIXTURE / "captions"),
                image_splits_dir=str(FIXTURE / "image_splits"),
                images_dir=str(images),
                artifacts_dir=str(root / "artifacts"),
                categories=("dress",),
            )
            with self.assertRaises(CoverageError):
                run_b0(cfg, dry_run=True, images_source="fixture")
            self.assertFalse((root / "artifacts" / "metrics.json").exists())

    def test_layout_discovery(self) -> None:
        layout = discover_fashioniq_layout(FIXTURE)
        self.assertIsNotNone(layout)
        assert layout is not None
        self.assertTrue((layout["captions_dir"] / "cap.dress.val.json").is_file())


class HubGateTests(unittest.TestCase):
    def test_publish_disabled_by_default(self) -> None:
        hub = HubConfig(enabled=False, repo_id="user/test")
        self.assertFalse(publishing_enabled(hub, flag=False))
        self.assertFalse(publishing_enabled(hub, flag=True))
        os.environ.pop("STYLEWEAVE_HF_PUBLISH", None)
        with self.assertRaises(PublishDisabled):
            publish_eval_artifacts(
                hub=hub,
                artifacts_dir=Path("/tmp/does-not-matter"),
                flag=True,
            )

    def test_publish_refuses_images(self) -> None:
        hub = HubConfig(enabled=True, repo_id="user/test", allow_images=True)
        os.environ["STYLEWEAVE_HF_PUBLISH"] = "1"
        try:
            with self.assertRaises(PublishRefused):
                publish_eval_artifacts(
                    hub=hub,
                    artifacts_dir=Path("/tmp/does-not-matter"),
                    flag=True,
                )
        finally:
            os.environ.pop("STYLEWEAVE_HF_PUBLISH", None)

    def test_query_schema_is_official(self) -> None:
        queries = load_queries(FIXTURE / "captions", "dress", "val")
        self.assertEqual(queries[0].reference_id, "IMG_A")
        self.assertEqual(queries[0].target_id, "IMG_B")
        self.assertIn(" and ", queries[0].text)


class MetricsArtifactContractTests(unittest.TestCase):
    def test_existing_metrics_must_declare_producer(self) -> None:
        artifacts = ROOT / "artifacts"
        if not artifacts.exists():
            return
        for path in artifacts.rglob("metrics.json"):
            payload = json.loads(path.read_text())
            self.assertEqual(payload.get("produced_by"), "styleweave.eval.b0")
            self.assertIn("metrics", payload)
            self.assertNotEqual(
                payload["metrics"].get("mean", {}).get("recall@10"),
                payload.get("literature_orientation", {}).get("mean_recall_at_10"),
            )


if __name__ == "__main__":
    unittest.main()
