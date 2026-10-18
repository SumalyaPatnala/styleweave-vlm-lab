# Phase 0 — Novelty assessment

**Do not claim research novelty in the README, UI, or paper draft.**
This document classifies proposed ideas against published work retrieved on
2026-10-18. “Experimentally distinct” means a local study design that is
worth running, not a new algorithm.

## Existing methods (do not present as new)

| Proposed idea | Already published as | Citation |
| --- | --- | --- |
| Frozen image–text encoder + image+text query | CLIP features; CLIP4Cir stage-0 / sum baseline | Radford et al. 2021; Baldrati et al. 2023 |
| Trainable image–text fusion module | TIRG; CLIP4Cir Combiner; CoSMo | Vo 2019; Baldrati 2023; Lee 2021 |
| Contrastive / ranking retrieval loss | InfoNCE CIR is standard | CLIP4Cir; TIRG; CoLLM |
| Prompt-based fusion | Pic2Word; SEARLE; SPRC | Saito 2023; Baldrati; Feng 2023 |
| LLM as the fusion backbone | CoLLM | Huynh et al., CVPR 2025 |
| Using fashion attributes with retrieval | FashionIQ side information | Wu et al., CVPR 2021 |
| VLM candidate verification | WISER retrieve–verify–refine | Wang et al., CVPR 2026 |
| Generative VLM + tool calling | Qwen3-VL visual agent (product capability) | Qwen team 2025 model card |
| Robustness / paraphrase / hard negatives | PinPoint; CIRCO; CoLLM refined splits | PinPoint 2026; CoLLM 2025 |
| Fine-grained attribute recognition | Fashionpedia | Jia et al., CVPR 2020 |

## Partially existing (combination, not invention)

| Idea | What already exists | What is only a design choice here |
| --- | --- | --- |
| Structured preserve / change / avoid | Relative captions already encode change; some conversational IR systems slot constraints | Making the three slots a first-class tensor / prompt field |
| Visual explanations for retrieval | Captioning and VLM rationales are common | Scoring **false** explanations against visible evidence |
| Attribute-conditioned Combiner | FashionIQ used attributes as side info; Combiners fuse image+text | Adding an optional attribute stream to a *small* frozen-CLIP Combiner |
| Local Mac MPS reproduction | Many papers assume NVIDIA GPUs | Documented MPS limits, parameter counts, and failure modes |

## Experimentally distinct (evaluation / systems study)

These are the only claims we are allowed to make after results exist, and
only as **empirical observations on this setup**:

1. A **controlled comparison** of (a) frozen CLIP sum, (b) CLIP-frozen
   Combiner, (c) Combiner + structured attribute slots, and (d) Qwen3-VL-2B
   tool-calling, on the **same FashionIQ val images, same preprocessing,
   same seeds, same hardware log**.
2. A **verified counterfactual subset** with human-checked preserve / change
   / avoid labels, held out from training, used to measure constraint
   satisfaction and explanation faithfulness — not just Recall@K.
3. An **honest trainable-module inventory** for any LoRA run (language
   adapters vs vision encoder vs merger), including a failed or skipped
   vision-encoder experiment if memory forbids it.

If those studies show a gain, the write-up will say “under this protocol,
on this hardware, this variant improved X.” It will not say “we propose a
novel architecture that advances SOTA.”

## Claims that are forbidden unless later evidence appears

- “State of the art on FashionIQ”
- “First method to use attributes / VLMs / constraints for CIR”
- “We fine-tuned the Qwen3-VL vision encoder” (false unless those tensors
  have `requires_grad=True` and were updated)
- Any numeric Recall, preference, or ablation result before the corresponding
  JSON artifact is written by an evaluation script
