# Phase 0 — Proposed experiment design

No experiment in this document has been executed. Numbers below are
**protocol**, not results.

## Research question (operationalized)

On FashionIQ validation, does adding (i) a learned fusion module and
(ii) explicit preserve / change / avoid attribute slots improve Recall@10
and Recall@50 over frozen CLIP late fusion, and do those gains coincide
with fewer constraint violations and fewer false visual explanations?

## Shared protocol (all retrieval systems)

- **Backbone:** `openai/clip-vit-base-patch32`, frozen unless a later
  document explicitly unfreezes it.
- **Images:** official FashionIQ IDs; RGB; CLIP processor (resize/crop/normalize).
- **Text:** official relative captions; CLIP tokenizer, max length 77.
- **Gallery:** official per-category validation image split.
- **Query:** reference image + relative caption (+ optional attribute slots).
- **Score:** cosine similarity between query embedding and gallery embeddings.
- **Metrics:** Recall@10, Recall@50, per category (dress, shirt, toptee) and
  mean. Exact definition: target is counted as a hit if its rank ≤ K.
- **Split:** train captions for any learned module; **val** for reporting.
  Test labels unpublished — do not scrape or guess them.
- **Seeds:** 13, 21, 42 for any stochastic training. Baseline retrieval is
  deterministic given weights and preprocessing.
- **Hardware log:** chip, memory, `torch` version, `mps` flag, wall time,
  peak process RSS.
- **Artifacts:** `artifacts/<run_id>/metrics.json` and `report.md` written
  only by evaluation code.

## Phase 1 — Frozen baseline

**Query representation (primary):**

```
q = normalize( encode_image(I_ref) ) + normalize( encode_text(T_mod) )
q = normalize(q)
```

**Controls:**

- Image-only: `normalize(encode_image(I_ref))`
- Text-only: `normalize(encode_text(T_mod))`

These are not “new methods”; they bound how much each modality contributes.

**Implementation gates:**

1. Loader validates official JSON schema and split disjointness
   (no train image ID in val split files; log multi-category IDs).
2. Missing image files are counted and fail the run above a configured
   threshold (default: any missing target or >1% missing gallery).
3. Unit tests for Recall@K on a 4-item synthetic ranking.
4. Integration test: encode ≥1 real FashionIQ image + caption and retrieve
   from a 16-image gallery.

**Stop condition:** if official images cannot be obtained with verifiable
IDs, do not train and do not fabricate a gallery.

## Phase 2 — Trainable fusion

**Module (`AttributeCombiner`), CLIP frozen:**

```
h_i = encode_image(I_ref)          # no grad
h_t = encode_text(T_mod)           # no grad
h_a = embed(attributes) or 0       # optional
z   = Combiner([h_i; h_t; h_a])    # trainable
```

Combiner: LayerNorm → Linear → GELU → Linear → L2 normalize.
Target: `normalize(encode_image(I_tgt))` with no grad.

**Loss:** in-batch InfoNCE (symmetric optional). Temperature initialized at
0.07 and learned or swept `{0.05, 0.07, 0.1}`.

**Ablations (same val protocol as Phase 1):**

| ID | Query |
| --- | --- |
| B0 | Frozen sum (Phase 1 primary) |
| B1 | Frozen image-only |
| B2 | Frozen text-only |
| M0 | Combiner(image, text) |
| M1 | Combiner(image, text, attributes) |

**Must log:** trainable parameter count, epoch loss curve, best val R@10,
seed, split hash, wall time.

Precompute CLIP embeddings to disk so training does not keep the encoder in
the optimizer graph.

## Phase 3 — Generative VLM (after Phase 1 artifacts)

Use `Qwen/Qwen3-VL-2B-Instruct` **only if** a dry-load of the processor +
model in float16 on MPS succeeds and peak RSS stays below ~14 GB.

Tasks (inference first):

1. Fine-grained attribute list for a reference image (JSON schema).
2. Parse a relative caption into `{preserve, change, avoid}`.
3. Generate a visual explanation for a retrieved pair
   `(I_ref, T_mod, I_cand)`.
4. Tool calling: `retrieve(query)` and `verify_attributes(image, slots)`.

**LoRA:** only if inference is stable. Language attention projections only;
`tune_mm_vision=False`. Print `requires_grad` inventory. If the process
OOMs, record the failure and stop. **Do not** fall back to claiming a
successful vision-encoder fine-tune.

Official Qwen `flash_attn` training is out of scope on this machine.

## Phase 4 — Original evaluation (verified subset)

Build `styleweave-cf-val` from FashionIQ **validation** captions only:

1. Automatically draft preserve / change / avoid slots from the caption and
   FashionIQ attribute vocabulary.
2. Human-verify a target of **100–200** queries (balanced by category).
3. Freeze the subset file; never train on it.
4. If a method needs early stopping, carve `val-dev` from remaining val
   captions — not from the counterfactual file.

**Metrics (in addition to R@K):**

| Metric | Definition |
| --- | --- |
| Preserve hit | Requested invariant attributes still present on the top-1 / top-10 candidate (human or FashionIQ labels) |
| Change hit | Requested modification present |
| Avoid violation | Forbidden attribute present |
| False explanation rate | Explanation asserts a visual fact absent from `I_cand` (human, binary) |
| Paraphrase delta | R@10 change across 3 caption paraphrases (inspired by PinPoint; we will not claim PinPoint-scale coverage) |
| Preference | Pairwise human preference on explanation groundedness, n small and reported as such |

This is an evaluation protocol adjacent to WISER / PinPoint, not a claim of
a new verifier.

## Phase 5 — Interface

Allowed only after Phase 1 `metrics.json` exists and is produced by code.
The UI must read those files. Hardcoded scores or canned explanations are
bugs.

## First executable experiment (not started)

**B0:** frozen CLIP ViT-B/32 L2-sum on official FashionIQ val. Closest
published analogue is CLIP4Cir Table 1 frozen sum (mean R@10/R@50
19.70 / 38.26). Do not start M0/M1 or Qwen until B0 writes `metrics.json`.

See canonical `docs/experiment_plan.md`.

## Reproducibility checklist

- [ ] Python 3.12 venv only
- [ ] Config YAML committed; run IDs include git commit + seed
- [ ] Data license text recorded at download time
- [ ] Missing-image report committed with metrics
- [ ] No push / deploy / paid API without approval
