# StyleWeave VLM Lab

Local, reproducible research on **composed fashion retrieval**: can explicit
visual attribute constraints and a small learned fusion module improve
FashionIQ retrieval while keeping explanations visually grounded?

Standalone research repository. Not WorldSense-AI and not a portfolio site.
Remote: **https://github.com/SumalyaPatnala/styleweave-vlm-lab** (`main` only).
Do not create a second repository.

## Current status

**Phase 0 documentation is complete. No model training.** Phase 1 B0 code is
in-repo; CLIP and FashionIQ **image** downloads are still gated. No metrics
until `styleweave.eval.b0` writes them.

Safer B0 path on this Mac: **Kaggle free GPU T4**, not local MPS. See
[docs/phase1_execution.md](docs/phase1_execution.md). Do not push until
approved.

| Phase | Status | Gate |
| --- | --- | --- |
| 0 Environment and literature audit | Complete (docs) | Review this packet before Phase 1 |
| 1 Frozen CLIP baseline on FashionIQ | Runner ready; full eval not run | Kaggle T4 preferred; coverage must pass before metrics.json |
| 2 Trainable fusion + contrastive training | Not started | Reproducible B0 |
| 3 Qwen3-VL-2B generative / tool-use | Deferred | Exceeds 5 GB cap unless re-approved |
| 4 Counterfactual + ablation evaluation | Not started | Train / val isolation |
| 5 Next.js research interface | Blocked | Real Phase 1 artifacts |

### Phase 0 packet (canonical)

1. [Hardware feasibility](docs/hardware_feasibility.md)
2. [Literature review](docs/literature_review.md)
3. [Research hypothesis](docs/research_hypothesis.md)
4. [Dataset audit](docs/dataset_audit.md)
5. [Model selection](docs/model_selection.md)
6. [Experiment plan](docs/experiment_plan.md)
7. [Resource budget](docs/resource_budget.md)

Supporting notes from the first audit live under `docs/phase0/`.
Re-verification log: [docs/phase0/07_REVERIFICATION.md](docs/phase0/07_REVERIFICATION.md).
Phase 1 runner: [docs/phase1_execution.md](docs/phase1_execution.md).
Kaggle notebook: [notebooks/kaggle_phase1_b0.ipynb](notebooks/kaggle_phase1_b0.ipynb).

## Repository path

```
/Users/sumalyapatnala/styleweave-vlm-lab
```

Python 3.12 virtualenv (numpy/PyYAML for tests; torch not installed locally):

```
/Users/sumalyapatnala/styleweave-vlm-lab/.venv/bin/python
```

## Hard constraints (Phase 0)

- Apple M3 Pro, **18 GB unified memory**, Metal / MPS only. No CUDA.
- About **32 Gi free disk** (2026-10-18 re-check). Existing Hugging Face cache is **67 GB** and has no CLIP or Qwen3-VL.
- Default `python3` on this Mac is 3.14. Use the repo 3.12 venv.
- FashionIQ official test labels are unpublished. Report on the official val split.
- Do not claim visual-encoder fine-tuning if only language LoRA is trained.
- Do not invent metrics, training curves, or model explanations.

## What is not novel

Frozen CLIP late fusion, Combiner-style fusion, contrastive CIR, prompt-based
pseudo-word methods, LLM fusion (CoLLM), and VLM retrieve-verify pipelines
(WISER) already exist. StyleWeave is a **controlled local study** of
attribute-conditioned fusion plus explanation faithfulness. See
[docs/phase0/03_NOVELTY_ASSESSMENT.md](docs/phase0/03_NOVELTY_ASSESSMENT.md).
