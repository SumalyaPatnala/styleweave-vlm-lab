# Phase 0 — Literature audit

Sources were retrieved on 2026-10-18 from official project pages, arXiv/ar5iv,
CVF Open Access, and Hugging Face model cards. This is a working review for
experiment design, not a claim of exhaustive coverage.

## Task definition

**Composed image retrieval (CIR):** given a reference image and a relative
caption, rank a gallery so the target that applies the requested change is
near the top. FashionIQ (Wu, Gao, Guo, Al-Halah, Rennie, Grauman, Feris;
CVPR 2021) is the standard fashion benchmark: dresses, shirts, tops & tees;
metrics Recall@10 and Recall@50. Official test labels are unpublished;
comparable papers report the validation split
([SEARLE issue discussion](https://github.com/miccunifi/SEARLE/issues/11)).

FashionIQ also ships product-derived **attribute side information** (DeepFashion
vocabulary: texture, fabric, shape, part, style). Using attributes is therefore
part of the original dataset paper, not a new problem statement.

## Method families

### 1. Late fusion / Combiner (supervised CIR)

- **TIRG** (Vo et al., CVPR 2019). Residual gating that composes image and
  text features. Canonical supervised CIR baseline.
- **CLIP4Cir** (Baldrati, Bertini, Uricchio, Del Bimbo; TOMM 2023). Task-adapt
  CLIP, then train a Combiner. Contrastive loss. Strong FashionIQ numbers.
  Code: https://github.com/ABaldrati/CLIP4Cir
- **CoSMo** (Lee, Kim, Han; CVPR 2021). Content-style modulation on FashionIQ.

**Relation to StyleWeave Phase 1–2:** the frozen CLIP sum and the trainable
fusion module are instances of this family. We should cite CLIP4Cir as the
closest engineering ancestor and treat our Combiner as a re-implementation
for controlled local comparison, not a new architecture.

### 2. Prompt / text-inversion fusion (mostly zero-shot CIR)

- **Pic2Word** (Saito et al., CVPR 2023; Google). Maps a picture to a
  pseudo-word token; compose with the modification text in CLIP text space.
  https://github.com/google-research/composed_image_retrieval
- **SEARLE / iSEARLE** (Baldrati et al.). Optimization + distilled inversion
  network for pseudo-word tokens.
- **SPRC** (Feng et al., 2023). Sentence-level prompts instead of a single
  pseudo-word; BLIP-2 captions + alignment losses.
  https://github.com/chunmeifeng/SPRC

**Relation:** StyleWeave does **not** start with text inversion. If a prompt
baseline is added later, SPRC/Pic2Word are the citations.

### 3. LLM / VLM fusion

- **CoLLM** (Huynh et al., CVPR 2025). Maps image features into an LLM,
  fuses with instruction-style text, projects to a retrieval embedding.
  Synthesizes triplets on the fly (Slerp + templates). Introduces MTCIR
  (3.4M) and refined CIRR / FashionIQ evaluation.
  Paper: https://arxiv.org/abs/2503.19910
  Project: https://collm-cvpr25.github.io

**Relation:** CoLLM is the closest *published* “use an LLM for CIR fusion”
method. StyleWeave Phase 2 is a **small MLP fusion**, not CoLLM. Phase 3
uses Qwen3-VL as a **generative / tool-calling** model, not as the retrieval
embedding backbone. Do not describe Phase 3 as CoLLM.

### 4. Robustness benchmarks and constraint verification

- **CIRCO** (used widely for ZS-CIR). Multiple annotated positives; mAP.
- **CoLLM refined benchmarks.** Reliability fixes on CIRR / FashionIQ.
- **PinPoint** (Mahadev, Yuan, Poirson, Xue, Wu, Kislyuk; CVPR 2026;
  arxiv:2603.04598). 7,635 queries, explicit hard negatives, six paraphrases
  per query, multi-image queries. Reports ~25% paraphrase variance and
  residual hard-negative retrieval.
- **WISER** (Wang, Qu, Yang, Hao, Xu, Guo, Wang; CVPR 2026;
  arxiv:2602.23029). Training-free retrieve–verify–refine. Dual-path
  T2I/I2I plus a VLM verifier of “instruction applied to reference.”
  Code: https://github.com/Physicsmile/WISER

**Relation:** Phase 4’s constraint checks and explanation-faithfulness tests
are in this evaluation family. WISER already does VLM verification of CIR
candidates. StyleWeave’s distinct angle is **structured preserve / change /
avoid slots** plus **false-explanation** measurement on a small verified
fashion subset — an evaluation protocol, not a new verifier algorithm.

### 5. Fashion attributes as a separate task

- **Fashionpedia** (Jia et al., CVPR 2020). 27 garment categories, 19 parts,
  294 attributes, masks. Official eval is AP with IoU and attribute F1.
  https://fashionpedia.github.io/home/

**Relation:** optional Phase 1/4 attribute evaluation. Fashionpedia images
are street / celebrity photos, not FashionIQ catalog images. Domain shift
must be stated. Do not treat Fashionpedia AP as a FashionIQ retrieval score.

## Comparison table

| Method | Year | Fusion | Training | FashionIQ protocol | Attributes / constraints | Explanations | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TIRG | 2019 | Residual gate | Supervised triplets | R@K | Implicit in text | No | No |
| FashionIQ paper model | 2021 | Transformer + attributes | Supervised + dialog | Interactive IR | Yes (side information) | No | No |
| CoSMo | 2021 | Content-style modulation | Supervised | R@10/50 | Implicit | No | No |
| CLIP4Cir Combiner | 2023 | Late Combiner on CLIP | Two-stage contrastive | R@10/50 val | Implicit | No | No |
| Pic2Word | 2023 | Pseudo-word prompt | ZS (image–caption) | R@10/50 | Implicit | No | No |
| SEARLE | 2023–24 | Optimized / distilled prompt | ZS | R@10/50 val | Implicit | No | No |
| SPRC | 2023 | Sentence-level prompt | Prompt + alignment | R@10/50 | Implicit | No | No |
| CoLLM | 2025 | LLM joint embedding | Synthetic + labeled | Refined CIR | Implicit / multi-text | No (not primary) | Benchmark refinement |
| WISER | 2026 | Dual-path + VLM verify | Training-free | R@10/50 | Instruction faithfulness | Self-reflection | Yes (VLM yes/no) |
| PinPoint | 2026 | Benchmark (many methods) | N/A | Broader CIR | Hard negatives, paraphrases | N/A | Explicit negatives |
| **StyleWeave (this lab)** | 2026 | Frozen CLIP + small fusion; Qwen3-VL tools | Local supervised fusion | Official val R@10/50 | Explicit preserve/change/avoid | Generated + scored | Planned, small verified set |

StyleWeave is listed for contrast, not as a published method.

## Survey pointer

Zhou et al., “A Survey on Composed Image Retrieval,” ACM TOMM 2025
(https://dl.acm.org/doi/10.1145/3723879) organizes CIR into supervised late
fusion, text inversion, training-free, and self-supervised families. Our
Phase 1–2 plan sits in supervised late fusion; Phase 3 sits near
training-free VLM verification.

## Citations (BibTeX)

See [../citations.bib](../citations.bib).
