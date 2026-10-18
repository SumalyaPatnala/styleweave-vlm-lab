# Literature review

Working review for StyleWeave VLM Lab. Sources were retrieved or
re-checked on **2026-10-18**. This is not an exhaustive survey and
does **not** claim a new algorithm.

Primary survey: Zhou et al., “A Survey on Composed Image Retrieval,” ACM
TOMM 2025. https://doi.org/10.1145/3723879

BibTeX: `docs/citations.bib`.

## Task

**Composed image retrieval (CIR):** rank a gallery given a reference image
and a relative caption. Fashion CIR uses FashionIQ (Wu et al., CVPR 2021).
Standard metrics are Recall@10 and Recall@50. Official test labels are
unpublished; later papers report the **validation** split
([SEARLE #11](https://github.com/miccunifi/SEARLE/issues/11)).

FashionIQ also ships product-derived **attribute side information**
(DeepFashion vocabulary: texture, fabric, shape, part, style). Using
attributes is part of the original dataset paper, not a new problem.

## Method families

### 1. Late fusion / Combiner (supervised CIR)

- **TIRG** (Vo et al., CVPR 2019). Residual gating of image and text.
- **CoSMo** (Lee, Kim, Han; CVPR 2021). Content–style modulation on FashionIQ.
- **CLIP4Cir** (Baldrati, Bertini, Uricchio, Del Bimbo; TOMM 2023).
  Task-adapt CLIP, then train a Combiner with contrastive loss.
  Code: https://github.com/ABaldrati/CLIP4Cir
  ACM: https://doi.org/10.1145/3617597 · arXiv: https://arxiv.org/abs/2308.11485

**Relation:** StyleWeave Phase 1–2 lives here. Frozen L2-sum (B0) and a
small Combiner (M0/M1) are re-implementations for a local comparison, not
new architectures. Closest engineering ancestor: CLIP4Cir.

### 2. Prompt / text-inversion (mostly zero-shot CIR)

- **Pic2Word** (Saito et al., CVPR 2023). Picture → pseudo-word token.
- **SEARLE** (Baldrati et al., ICCV 2023). Optimized / distilled inversion.
- **SPRC** (Feng et al., 2023). Sentence-level prompts.

**Relation:** not the Phase 1 starting point. Cite if a prompt baseline is
added later.

### 3. LLM / VLM fusion

- **CoLLM** (Huynh et al., CVPR 2025). Maps image features into an LLM,
  fuses with instruction-style text, projects to a retrieval embedding.
  https://arxiv.org/abs/2503.19910 · https://collm-cvpr25.github.io

**Relation:** CoLLM is the published “LLM as CIR fusion backbone.”
StyleWeave Phase 2 is a **small MLP**. Phase 3 (if ever loaded) uses
Qwen3-VL as a **parser / explainer / tool caller**, not as the retrieval
embedding backbone. Do not describe Phase 3 as CoLLM.

### 4. Robustness benchmarks and constraint verification

- **CIRCO**. Multiple annotated positives; mAP.
- **WISER** (Wang, Qu, Yang, Hao, Xu, Guo, Wang; CVPR 2026).
  Training-free retrieve–verify–refine. Dual-path T2I/I2I plus a VLM
  verifier of “instruction applied to reference.”
  https://arxiv.org/abs/2602.23029 · https://github.com/Physicsmile/WISER
  CVF: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_WISER_Wider_Search_Deeper_Thinking_and_Adaptive_Fusion_for_Training-Free_CVPR_2026_paper.html
- **PinPoint** (Mahadev, Yuan, Poirson, Xue, Wu, Kislyuk; CVPR 2026).
  7,635 queries, explicit hard negatives, six paraphrases per query.
  Reports ~25% paraphrase variance and residual hard-negative retrieval.
  https://arxiv.org/abs/2603.04598

**Relation:** Phase 4 constraint checks and false-explanation scoring sit
in this **evaluation** family. WISER already verifies CIR candidates with
a VLM. StyleWeave’s local angle is structured preserve / change / avoid
slots plus human false-explanation labels on a small verified fashion
subset — a protocol, not a new verifier.

### 5. Fashion attributes as a separate task

- **Fashionpedia** (Jia et al., CVPR 2020). 27 garment categories, 19 parts,
  294 localized attributes. Official eval is AP with IoU and attribute F1.
  https://fashionpedia.github.io/home/

**Relation:** optional later attribute evaluation. Images are street /
celebrity photos, not FashionIQ catalog shots. Domain shift must be
stated. Do not treat Fashionpedia AP as a FashionIQ retrieval score.

## Method comparison

| Method | Year | Paper | Fusion | Supervision | FashionIQ | Constraints / attributes | Explanations | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TIRG | 2019 | https://arxiv.org/abs/1812.07119 | Residual gate | Labeled triplets | Used in later comparisons | Implicit in text | No | No |
| FashionIQ | 2021 | https://arxiv.org/abs/1905.12794 | Transformer + side info | Supervised / dialog | Introduces the benchmark | Product attributes | No | No |
| CoSMo | 2021 | https://arxiv.org/abs/2104.03019 | Content–style modulation | Supervised | Yes | Implicit | No | No |
| CLIP | 2021 | https://arxiv.org/abs/2103.00020 | Dual encoder | Image–text web pairs | Encoder only | None | No | No |
| CLIP4Cir | 2023 | https://doi.org/10.1145/3617597 | Late Combiner on CLIP | Two-stage contrastive | R@10/50 **val** | Implicit | No | No |
| Pic2Word | 2023 | https://arxiv.org/abs/2302.03084 | Pseudo-word token | Zero-shot (captions) | Yes | Implicit | No | No |
| SEARLE | 2023 | https://arxiv.org/abs/2303.15247 | Optimized / distilled prompt | Zero-shot | Val protocol | Implicit | No | No |
| SPRC | 2023 | https://arxiv.org/abs/2310.05473 | Sentence-level prompt | Prompt + alignment | Yes | Implicit | No | No |
| Fashionpedia | 2020 | https://arxiv.org/abs/2004.12276 | N/A (attribute localization) | Detection / F1 | Different images | 294 localized attributes | No | Official FPEval |
| SigLIP 2 | 2025 | https://arxiv.org/abs/2502.14786 | Dual encoder | Web-scale sigmoid loss | Not a CIR method | None | No | No |
| CoLLM | 2025 | https://arxiv.org/abs/2503.19910 | LLM joint embedding | Synthetic + labeled CIR | Refined eval | Multi-text / implicit | Not primary | Benchmark refinement |
| Qwen3-VL | 2025 | https://arxiv.org/abs/2505.09388 | Native VLM | Instruction / agent | Not CIR-specific | Tool calling | Generative | Product capability |
| WISER | 2026 | https://arxiv.org/abs/2602.23029 | Dual-path + VLM verify | Training-free | R@10/50 | Instruction faithfulness | Self-reflection | Yes (VLM yes/no) |
| PinPoint | 2026 | https://arxiv.org/abs/2603.04598 | Benchmark | N/A | Broader CIR | Hard negatives, paraphrases | N/A | Explicit negatives |

## Published FashionIQ val numbers we may cite (not our results)

CLIP4Cir Table 1 (Baldrati et al., TOMM 2023 / arXiv:2308.11485), FashionIQ
**validation**, CLIP features. These are **their** numbers. Do not copy
them into `artifacts/` or call them StyleWeave metrics.

| CLIP4Cir condition | Shirt R@10/50 | Dress R@10/50 | Toptee R@10/50 | Mean R@10/50 |
| --- | --- | --- | --- | --- |
| Frozen sum (no encoder FT) | 19.53 / 35.57 | 17.70 / 36.29 | 21.88 / 42.93 | **19.70 / 38.26** |
| Frozen Combiner (no encoder FT) | 31.85 / 52.50 | 27.22 / 50.62 | 33.81 / 57.57 | **30.96 / 53.56** |
| Both encoders FT + Combiner | 39.87 / 60.84 | 37.67 / 63.16 | 44.88 / 68.59 | **40.80 / 64.20** |

StyleWeave **B0** is the local analogue of the frozen-sum row. StyleWeave
**M0** is the local analogue of the frozen Combiner row. Encoder
fine-tuning (the 40.80 row) is **out of scope** on this 18 GB Mac.

A local B0 that lands far from 19.70 is a protocol or data-coverage
issue to diagnose, not a claim that we beat CLIP4Cir. Preprocessing,
CLIP checkpoint, missing images, and caption pairing can all move R@K.

## What is already solved in the literature

1. Late fusion of CLIP image and text for FashionIQ (CLIP4Cir, TIRG, CoSMo).
2. Prompt / text-inversion CIR without labeled triplets (Pic2Word, SEARLE, SPRC).
3. LLM-as-fusion-backbone (CoLLM, CVPR 2025).
4. Attribute side information on FashionIQ (Wu et al., 2021).
5. VLM verification that a candidate matches “instruction applied to
   reference” (WISER, CVPR 2026).
6. Robustness evaluation with paraphrases and hard negatives (PinPoint; CIRCO).

## Experimentally open (not claimed as novel methods)

Published work measures Recall@K, and some recent work measures instruction
faithfulness or paraphrase variance. A **small, human-verified fashion
subset** that jointly scores retrieval, explicit preserve / change / avoid
slot satisfaction, and false visual explanations, under **identical**
frozen-CLIP, Combiner, and (if hardware allows) Qwen tool-use conditions,
is a local study worth running. It is a protocol adjacent to WISER and
PinPoint, not a new fusion algorithm.

See `docs/research_hypothesis.md` and `docs/phase0/03_NOVELTY_ASSESSMENT.md`.
