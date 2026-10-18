# Experiment plan

No training or checkpoint download is authorized in Phase 0. Numbers from
other papers are **literature orientation**, not StyleWeave results.

## Tracks

### Track A — retrieval

Frozen `openai/clip-vit-base-patch32`. FashionIQ official **val**.
Query: reference image + relative caption. Score: cosine.

| ID | Query | Trainable | CLIP4Cir analogue (published val mean R@10/R@50) |
| --- | --- | --- | --- |
| **B0** | `normalize(img) + normalize(text)` then L2 | None | Frozen sum **19.70 / 38.26** |
| B1 | image only | None | Control (not a CIR method) |
| B2 | text only | None | Control |
| M0 | Combiner(img, text) | Combiner only | Frozen Combiner **30.96 / 53.56** |
| M1 | Combiner(img, text, attribute slots) | Combiner + slot embed | No published twin |

CLIP4Cir’s best row (both encoders fine-tuned + Combiner, 40.80 / 64.20)
is **out of scope**: we will not unfreeze CLIP.

Loss for M0/M1: in-batch InfoNCE, temperature 0.07 (sweep 0.05/0.07/0.1).
Precompute CLIP embeddings to disk so the optimizer never sees the encoder.
Seeds: 13, 21, 42.

Metrics: Recall@10, Recall@50, per dress / shirt / toptee and mean.
A query is a hit at K iff the labeled target ranks ≤ K (single positive).
Attribute-aware extras (Track C) on the verified subset only.

### Track B — generative VLM

**Phase 0 work only:** feasibility, licenses, memory table
(`docs/hardware_feasibility.md`, `docs/model_selection.md`).

Later, if approved: Qwen3-VL-2B for (1) attribute JSON, (2)
preserve/change/avoid parse, (3) visual explanation, (4) tool calls to
`retrieve` and `verify_attributes`. LoRA is **deferred**. Official
NVIDIA recipe is out of scope.

### Track C — methodology

1. Literature novelty audit (this packet): do not claim first-ness.
2. Explicit preserve / change / avoid slots (text schema + optional tensor).
3. Baselines B0–B2 and ablations M0/M1 under one protocol.
4. Grounded explanations: human false-explanation rate on top-1.
5. Counterfactual subset `styleweave-cf-val` (100–200 val queries,
   human-checked, never trained on).
6. Failure analysis: missing-image report, per-category errors, paraphrase
   delta (3 paraphrases; not PinPoint-scale).

## Shared gates

- Python 3.12.7 repo `.venv` only. No second environment.
- Real FashionIQ images and official `captions/` + `image_splits/` JSON.
  Missing target → fail the run.
- `artifacts/<run_id>/metrics.json` and `report.md` written only by code.
- Hardware log: chip, memory, torch, MPS flag, wall time, peak RSS.
- Train / val isolation: Combiner sees train captions; report on val;
  `styleweave-cf-val` is a frozen slice of val, not used for early stopping
  (carve `val-dev` from remaining val if needed).
- No FashionIQ test labels.

## Recommended first baseline (after approval)

**B0 on FashionIQ val with frozen CLIP ViT-B/32.**

This is the smallest experiment that can fail honestly: no training, one
encoder, official val protocol, comparable to CLIP4Cir’s frozen-sum row.

### Why B0 first

1. It bounds what frozen CLIP already knows before any Combiner is trained.
2. It is deterministic given weights and preprocessing (seed 42 is for
   logging only).
3. CLIP weights are 605 MB; official text is 7 MB. Images are the only
   large remaining fetch, and they are required for every later run.
4. M0/M1 training on a broken loader would waste the disk budget.

### Steps after Phase 1 is approved (still not started)

1. Re-check `df -h /System/Volumes/Data` (need ≥25 Gi free **after** the
   planned peak). Last check: **32 Gi** free (2026-10-18).
2. Write a size estimate into the run ticket:
   CLIP 605,247,071 B; official JSON 7,146,567 B; images only if a mirror
   HEAD is acceptable and peak extract stays inside the 5 GB cap **and**
   the 25 Gi free floor.
3. Download official `captions/` + `image_splits/` first. Dry-run ID
   coverage against split files **before** extracting images.
4. Download CLIP-B/32 into the **project** cache path (do not assume the
   67 GB global HF cache will be reused; it currently has no CLIP).
5. Implement loader, CLIP preprocess, frozen encode, ranking, R@10/R@50.
6. Emit `artifacts/phase1_frozen_clip/metrics.json` + `report.md`. Run
   unit + integration tests.

Do not start M0/M1 training until B0 is reproducible. Do not download
Qwen, SigLIP2, or Fashionpedia for B0.

### Phase 1 success criteria (reproducibility, not SOTA)

- Official val JSON present; split hashes logged.
- Missing-target count = 0, or the run fails.
- Gallery missing fraction ≤ 1%, or the run fails.
- `metrics.json` produced by evaluation code (no hand-edited scores).
- Per-category and mean R@10 / R@50 present.
- Hardware log present.
- Write-up cites CLIP4Cir frozen-sum 19.70 / 38.26 as orientation and
  **does not** claim to beat CLIP4Cir if we only ran frozen B0.

## Phase 5 UI

Blocked until B0 `metrics.json` exists. No hardcoded scores.

## Stop / go

| Condition | Action |
| --- | --- |
| Phase 0 (this packet) | Documentation complete. |
| Phase 1 B0 runner | Implemented. Prefer Kaggle T4. Local MPS is the disk-risk fallback. |
| Phase 1 B0 metrics.json | Written only by `styleweave.eval.b0` after coverage passes. |
| Disk would fall below 25 GB free | Do not download |
| Image mirror missing >1% val IDs | Stop; do not train |
| Qwen dry-load RSS > ~14 GB | Skip Track B weights |
