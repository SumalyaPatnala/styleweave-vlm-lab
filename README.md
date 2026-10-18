# StyleWeave VLM Lab

Local, reproducible research on **composed fashion retrieval**: can explicit
visual attribute constraints and a small learned fusion module improve
FashionIQ retrieval while keeping explanations visually grounded?

This repository is **standalone**. It is not WorldSense-AI and it is not a
portfolio site. Nothing here has been pushed or deployed.

## Current status

**Phase 0 is complete. No model training has been run. No metrics exist yet.**

| Phase | Status | Gate |
| --- | --- | --- |
| 0 Environment and literature audit | Complete (this commit) | Report before training |
| 1 Frozen CLIP baseline on FashionIQ | Not started | Real images and official splits |
| 2 Trainable fusion + contrastive training | Not started | Identical eval vs Phase 1 |
| 3 Qwen3-VL-2B generative / tool-use | Not started | Hardware permit check |
| 4 Counterfactual + ablation evaluation | Not started | Train / val / test isolation |
| 5 Next.js research interface | Blocked | Requires real Phase 1 artifacts |

Read the Phase 0 packet before implementing anything:

1. [Executive summary](docs/phase0/00_EXECUTIVE_SUMMARY.md)
2. [Environment](docs/phase0/01_ENVIRONMENT.md)
3. [Literature](docs/phase0/02_LITERATURE.md)
4. [Novelty assessment](docs/phase0/03_NOVELTY_ASSESSMENT.md)
5. [Datasets and licenses](docs/phase0/04_DATASETS_AND_LICENSES.md)
6. [Model and hardware feasibility](docs/phase0/05_MODEL_HARDWARE.md)
7. [Experiment design](docs/phase0/06_EXPERIMENT_DESIGN.md)

## Repository path

```
/Users/sumalyapatnala/styleweave-vlm-lab
```

Python 3.12 virtualenv (created, packages not installed):

```
/Users/sumalyapatnala/styleweave-vlm-lab/.venv/bin/python
```

## Hard constraints (Phase 0)

- Apple M3 Pro, **18 GB unified memory**, Metal / MPS only. No CUDA.
- About **33 GB free disk**. Existing Hugging Face cache is **67 GB**.
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
