# Model selection

Phase 0 does **not** download checkpoints. Sizes below were re-checked
from Hugging Face API metadata and HTTP `x-linked-size` on **2026-10-18**.
No weight bodies were downloaded. The local Hugging Face hub cache
(`~/.cache/huggingface`, 67 GB) contains **no CLIP, SigLIP2, or Qwen3-VL**
repos.

## Track A — retrieval encoder (choose one to download later)

### Primary: OpenAI CLIP ViT-B/32

| Field | Value |
| --- | --- |
| ID | `openai/clip-vit-base-patch32` |
| Card | https://huggingface.co/openai/clip-vit-base-patch32 |
| Paper | Radford et al., ICML 2021. https://arxiv.org/abs/2103.00020 |
| Code license | MIT (https://github.com/openai/CLIP) |
| Weights | `pytorch_model.bin` **605,247,071 B** (`x-linked-size`, commit `3d74acf9`) |
| Architecture | ViT-B/32 image encoder + Transformer text encoder, 512-d |
| Transformers class | `CLIPModel` + `CLIPProcessor` |
| Inference memory | HF discussion FP16 ~289 MB weights; process **~0.6–1.5 GB** (estimate, unmeasured here) |
| CIR precedent | CLIP4Cir (2023) frozen-sum row is the B0 analogue |
| Why first | Smallest serious CIR encoder; fits the 5 GB experiment cap with FashionIQ text + a later image fetch |

CLIP card (re-read 2026-10-18): unconstrained **deployment is out of
scope**. Non-deployed image search is recommended only with thorough
in-domain testing and a **fixed class taxonomy**. This lab is constrained
research image search on FashionIQ categories (dress / shirt / toptee),
English captions only. Surveillance and face recognition remain out of
scope regardless of accuracy.

### Deferred alternative: SigLIP 2 Base (patch16, 224)

| Field | Value |
| --- | --- |
| ID | `google/siglip2-base-patch16-224` |
| Card | https://huggingface.co/google/siglip2-base-patch16-224 |
| Paper | Tschannen et al., 2025. https://arxiv.org/abs/2502.14786 |
| License | Apache-2.0 |
| Params | 375,187,970 F32 (HF `safetensors` metadata) |
| Weights | `model.safetensors` **1,500,800,904 B** (`x-linked-size`, commit `75de2d55`) |
| Why not first | ~2.5× CLIP on disk; no FashionIQ Combiner literature as established as CLIP4Cir |

Use SigLIP2 only as a **later frozen-encoder ablation**, after CLIP
baseline artifacts exist and disk is re-checked. Do not download both
CLIP and SigLIP2 at once.

### Rejected for this Mac (Track A)

- CLIP ViT-L/14, SigLIP2-so400m, SigLIP2-giant: quality vs memory/disk.
- Unfreezing the vision encoder: not in the 18 GB training budget.

## Track B — generative VLM (download deferred)

### Candidate: Qwen3-VL-2B-Instruct

| Field | Value |
| --- | --- |
| ID | `Qwen/Qwen3-VL-2B-Instruct` |
| Card | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct |
| Code | https://github.com/QwenLM/Qwen3-VL |
| Paper | Qwen3 Technical Report, 2025. https://arxiv.org/abs/2505.09388 |
| License | **Apache-2.0** (model-card YAML, re-checked 2026-10-18) |
| Params | 2,127,532,032 BF16 |
| Weights | `model.safetensors` **4,255,140,312 B** (`x-linked-size`, commit `89644892`) |
| Transformers | Card historically required ≥4.57; class `Qwen3VLForConditionalGeneration` |
| Documented extras | Visual agent / tool calling; FA2 recommended for multi-image |
| Official train extras | `flash_attn`, DeepSpeed — **not on MPS** |

Third-party (non-official) memory estimates: ~4.6–6.7 GB FP16 inference;
~7 GB LoRA; ~19 GB full FT. On 18 GB unified these are **upper-bound
warnings**, not measurements. Peak RSS on this Mac is **unverified**.

**Decision:** paper-only in Phase 0. Download only after Phase 1 CLIP
results exist, a written size estimate is approved, and `df` still shows
≥25 GB free **after** 4.26 GB + ~1 GB HF temp. That fetch **exceeds the
5 GB experiment cap** unless the cap is explicitly raised.

### Fallback if HF+MPS fails

`Qwen/Qwen3-VL-2B-Instruct-GGUF` (Metal / llama.cpp). Still a multi-GB
download — same approval gate. Not size-checked in this pass.

### Smaller VLMs

Only justified if Qwen3-VL-2B cannot load and Track C still needs
structured JSON. Re-audit then. Do not fetch alternatives speculatively.

## Track A fusion module (no pretrained weights)

`AttributeCombiner`: LayerNorm → Linear → GELU → Linear, ~1–3M parameters,
trained from scratch on frozen embeddings. This is a CLIP4Cir-style
Combiner, not a new architecture.

## Trainable-module rule

Any later PEFT run must print modules with `requires_grad=True`. If only
language LoRA tensors update, the report says **language-layer adapters
only**. Default Qwen flags `tune_mm_vision`, `tune_mm_mlp`, `tune_mm_llm`
are all false until set
(https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/train/argument.py).
