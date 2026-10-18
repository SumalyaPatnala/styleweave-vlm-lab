# Research hypothesis

## Core question

Can **explicit visual attribute constraints** (preserve / change / avoid)
and a **small learned multimodal fusion module** improve compositional
fashion retrieval on FashionIQ while keeping explanations **visually
grounded**?

This question is **empirical**, not a novelty claim. Fusion and attributes
already exist (CLIP4Cir; Wu et al. 2021). VLM verification already exists
(WISER 2026). The hypothesis is that a *joint* local protocol can show
whether retrieval gains and groundedness move together.

## Hypotheses (pre-registered; untested)

**H1 (retrieval).** On official FashionIQ validation, a CLIP-frozen
Combiner (M0) has higher mean Recall@10 than frozen L2-sum (B0).

Literature orientation, not a pass/fail number: CLIP4Cir reports frozen
sum 19.70 and frozen Combiner 30.96 mean R@10 on FashionIQ val. Encoder
fine-tuning is out of scope here, so H1 is **M0 vs B0 under our protocol**,
not “match CLIP4Cir’s 40.80 FT row.”

**H2 (attributes).** Adding structured preserve / change / avoid slots to
the Combiner (M1) does not reduce mean Recall@10 versus M0, and reduces
avoid-violations on a held-out verified subset.

**H3 (grounding).** Explanations produced for top-1 pairs have a lower
false-explanation rate when the retriever is M1 than when it is B0, on the
same verified subset. “False” means an asserted visual fact that is not
supported by the candidate image (human binary label).

**H4 (VLM, conditional).** If Qwen3-VL-2B inference fits this Mac, its
tool-use rerank does not need to beat M1 on Recall@K to be useful: it is
evaluated as a **constraint parser and explanation generator**, not as a
replacement encoder.

## What would falsify the study (not the field)

- M0 ≤ B0 on mean R@10 across three seeds, with overlapping uncertainty.
- M1 increases avoid-violations versus M0.
- Explanation “gains” disappear when raters are blinded to the system name.
- Any numeric claim without a `metrics.json` written by evaluation code.

## What we will not claim even if H1–H3 hold

- State of the art on FashionIQ.
- First use of attributes, VLMs, or constraints for CIR.
- Visual-encoder fine-tuning unless those tensors were updated.
- Results on FashionIQ **test** (labels unpublished).

## Track mapping

| Track | Hypothesis | Primary citations |
| --- | --- | --- |
| A Retrieval | H1, H2 | CLIP (2021), CLIP4Cir (2023), FashionIQ (2021) |
| B Generative VLM | H4 | Qwen3-VL (2025), WISER (2026) |
| C Methodology | H2, H3 | FashionIQ attributes; WISER; PinPoint (2026) |

## First measurement that can test anything

B0 must exist before H1–H3 can be tested. B0 does **not** test H1; it
creates the frozen-sum reference that H1 compares against.
