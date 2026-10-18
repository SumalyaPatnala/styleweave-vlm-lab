# Phase 0 — Executive summary

**Date of audit:** 2026-10-18
**Machine:** MacBook Pro (Mac15,6), Apple M3 Pro, 18 GB unified memory
**Repository:** `/Users/sumalyapatnala/styleweave-vlm-lab` (isolated)
**Remote:** https://github.com/SumalyaPatnala/styleweave-vlm-lab (`main`)
**Isolation:** WorldSense-AI and portfolio repositories were inspected, not modified
**Training status:** none
**Canonical packet:** `docs/*.md`. Re-check log: `docs/phase0/07_REVERIFICATION.md`.

## Verdict

Phase 1 (frozen CLIP baseline) and Phase 2 (small trainable fusion) are
**feasible on this machine**. Phase 3 Qwen3-VL-2B **inference** is
conditionally feasible. Official Qwen LoRA training is **not portable** to
MPS. Fashionpedia is **optional and deferred** until disk is freed.

No research novelty is claimed. Several proposed ideas already exist as
published methods. The experimentally distinct part is a **joint evaluation
protocol**: retrieval, constraint satisfaction, and explanation faithfulness
on a verified counterfactual fashion subset, compared under identical local
conditions.

## Go / no-go

| Workstream | Decision | Reason |
| --- | --- | --- |
| Phase 1 frozen CLIP + FashionIQ R@10/R@50 | **Go** | CLIP-B/32 is ~605 MB; MPS supported; official val protocol is standard |
| Phase 2 small fusion + InfoNCE | **Go** | Train CLIP-frozen combiner; ~1–3M trainable params |
| Phase 3 Qwen3-VL-2B inference / tool-use | **Conditional go** | 4.26 GB BF16 weights; 18 GB unified + swap already high |
| Phase 3 official LoRA recipe | **No** | Requires FlashAttention / DeepSpeed / NVIDIA stack |
| Phase 3 language-only PEFT on a tiny split | **Marginal** | Try only after inference fits; log trainable modules |
| Phase 3 vision-encoder updates | **No on this hardware** | Do not claim it |
| Full Fashionpedia image download | **Defer** | Train images are 3.34 GB; annotations-only is enough later |
| Phase 5 Next.js UI | **Blocked** | Requires real Phase 1 evaluation artifacts |
| Push / deploy / paid GPU | **Blocked** | Explicitly not approved |

## Binding constraints

1. **Disk.** Data volume has **32 Gi** free (re-check) and a 67 GB Hugging
   Face cache with no CLIP or Qwen3-VL. FashionIQ community rar is
   988,840,917 B. Qwen3-VL-2B is 4,255,140,312 B. Do not download unused
   models.
2. **Memory.** 18 GB unified, Metal 4, 14 GPU cores. The machine is already
   swapping. Close other apps before any VLM load.
3. **Python.** System `python3` is 3.14.6. Project interpreter is Homebrew
   **Python 3.12.7** in `.venv`.
4. **No CUDA.** FlashAttention-2 is unavailable. MPS fallback only.
5. **FashionIQ test GT is unpublished.** Comparable papers report validation
   Recall@10 / Recall@50. We will do the same and say so.

## What happens next (not started)

Phase 1 (not started; needs approval) should run **B0**: frozen CLIP-B/32
L2-sum on official FashionIQ val. Download official `captions/` +
`image_splits/` first, then CLIP, then one documented image mirror after
an ID dry-run. No Combiner training until that baseline is reproducible.
