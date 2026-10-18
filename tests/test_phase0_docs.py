"""Phase 0 gates: docs exist, isolation paths are outside this repo."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = [
    "README.md",
    "NOTICE",
    "docs/phase0/00_EXECUTIVE_SUMMARY.md",
    "docs/phase0/01_ENVIRONMENT.md",
    "docs/phase0/02_LITERATURE.md",
    "docs/phase0/03_NOVELTY_ASSESSMENT.md",
    "docs/phase0/04_DATASETS_AND_LICENSES.md",
    "docs/phase0/05_MODEL_HARDWARE.md",
    "docs/phase0/06_EXPERIMENT_DESIGN.md",
    "docs/citations.bib",
    "configs/experiments/phase1_frozen_clip.yaml",
    "configs/experiments/phase2_combiner.yaml",
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
        metric_files = list(artifacts.rglob("metrics.json"))
        self.assertEqual(
            metric_files,
            [],
            "metrics.json must be created by an evaluation run, not Phase 0: "
            f"{metric_files}",
        )

    def test_phase0_docs_forbid_novelty_claim(self) -> None:
        text = (ROOT / "docs/phase0/03_NOVELTY_ASSESSMENT.md").read_text()
        self.assertIn("Do not claim research novelty", text)
        self.assertIn("Existing methods", text)


if __name__ == "__main__":
    unittest.main()
