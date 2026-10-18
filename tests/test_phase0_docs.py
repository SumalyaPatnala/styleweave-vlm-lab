"""Phase 0 gates: docs exist, isolation paths are outside this repo."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = [
    "README.md",
    "NOTICE",
    "docs/hardware_feasibility.md",
    "docs/literature_review.md",
    "docs/research_hypothesis.md",
    "docs/dataset_audit.md",
    "docs/model_selection.md",
    "docs/experiment_plan.md",
    "docs/resource_budget.md",
    "docs/phase0/00_EXECUTIVE_SUMMARY.md",
    "docs/phase0/01_ENVIRONMENT.md",
    "docs/phase0/02_LITERATURE.md",
    "docs/phase0/03_NOVELTY_ASSESSMENT.md",
    "docs/phase0/04_DATASETS_AND_LICENSES.md",
    "docs/phase0/05_MODEL_HARDWARE.md",
    "docs/phase0/06_EXPERIMENT_DESIGN.md",
    "docs/phase0/07_REVERIFICATION.md",
    "docs/citations.bib",
    "configs/experiments/phase1_frozen_clip.yaml",
    "configs/experiments/phase2_combiner.yaml",
    "docs/phase1_execution.md",
    "notebooks/kaggle_phase1_b0.ipynb",
]


class Phase0DocsTests(unittest.TestCase):
    def test_required_phase0_docs_exist(self) -> None:
        missing = [rel for rel in REQUIRED_DOCS if not (ROOT / rel).is_file()]
        self.assertEqual(missing, [], f"Missing Phase 0 files: {missing}")

    def test_repo_name_and_isolation(self) -> None:
        self.assertEqual(ROOT.name, "styleweave-vlm-lab")
        self.assertEqual(
            ROOT.resolve(),
            Path("/Users/sumalyapatnala/styleweave-vlm-lab"),
        )
        self.assertFalse((ROOT / "WorldSense-AI").exists())

    def test_no_fabricated_metrics_artifacts(self) -> None:
        artifacts = ROOT / "artifacts"
        if not artifacts.exists():
            return
        for path in artifacts.rglob("metrics.json"):
            payload = path.read_text()
            self.assertIn("styleweave.eval.b0", payload)
            self.assertIn("produced_by", payload)

    def test_phase0_docs_forbid_novelty_claim(self) -> None:
        text = (ROOT / "docs/phase0/03_NOVELTY_ASSESSMENT.md").read_text()
        self.assertIn("Do not claim research novelty", text)
        self.assertIn("Existing methods", text)

    def test_canonical_literature_review_is_sourced(self) -> None:
        text = (ROOT / "docs/literature_review.md").read_text()
        self.assertIn("CLIP4Cir", text)
        self.assertIn("19.70", text)
        self.assertIn("does **not** claim a new algorithm", text.replace("\n", " "))
        self.assertIn("WISER", text)
        self.assertIn("PinPoint", text)
        self.assertNotIn("state of the art", text.lower())

    def test_dataset_audit_records_license_gap_and_official_paths(self) -> None:
        text = (ROOT / "docs/dataset_audit.md").read_text()
        self.assertIn("captions/", text)
        self.assertIn("image_splits/", text)
        self.assertIn("issue #22", text)
        self.assertIn("988,840,917", text)
        self.assertIn("Never commit", text)
        self.assertIn("does **not**\ndownload archives", text)

    def test_experiment_plan_recommends_b0_first(self) -> None:
        text = (ROOT / "docs/experiment_plan.md").read_text()
        self.assertIn("B0 on FashionIQ val with frozen CLIP ViT-B/32", text)
        self.assertIn("Do not start M0/M1 training until B0 is reproducible", text)
        self.assertIn("40.80", text)
        self.assertIn("out of scope", text)

    def test_model_selection_defers_qwen_download(self) -> None:
        text = (ROOT / "docs/model_selection.md").read_text()
        self.assertIn("4,255,140,312", text)
        self.assertIn("605,247,071", text)
        self.assertIn("Apache-2.0", text)
        self.assertIn("does **not** download checkpoints", text)

    def test_readme_points_at_existing_github_remote(self) -> None:
        text = (ROOT / "README.md").read_text()
        self.assertIn("https://github.com/SumalyaPatnala/styleweave-vlm-lab", text)
        self.assertIn("Do not create a second repository", text)
        self.assertIn("Phase 0 documentation is complete", text)
        self.assertIn("No model training", text)
        self.assertIn("docs/phase1_execution.md", text)


if __name__ == "__main__":
    unittest.main()
