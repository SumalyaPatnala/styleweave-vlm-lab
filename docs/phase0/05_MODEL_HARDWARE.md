# Phase 0 — Model and hardware feasibility

Official documentation checked on 2026-10-18.

## Frozen encoder: CLIP ViT-B/32

- Card: https://huggingface.co/openai/clip-vit-base-patch32
- Architecture: ViT-B/32 image encoder + Transformer text encoder, 512-d
- Weights: **605,247,071 B** (`pytorch_model.bin`, HF `x-linked-size` 2026-10-18)
- HF memory utility (discussion #11): FP16 ~289 MB weights; FP32 ~577 MB
- Transformers API: `CLIPModel`, `CLIPProcessor`
- Apple Silicon: supported via PyTorch MPS after install

**Feasible for Phase 1 and 2.** This is the default backbone.

Larger CLIP (ViT-L/14) is deferred. It would improve quality and increase
gallery-embedding time and memory; 18 GB unified is enough for inference but
leaves less headroom for Qwen in the same session.

## Generative VLM: Qwen3-VL-2B-Instruct

Official sources:

- Model card: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- Code: https://github.com/QwenLM/Qwen3-VL
- License: Apache-2.0
- Weights: 2,127,532,032 BF16 params, `model.safetensors` **4,255,140,312 B**
- Transformers class: `Qwen3VLForConditionalGeneration`
- Card install note (historical): `transformers>=4.57` / install from source
  when 4.57 was unreleased. PyPI now has 4.57.6 and 5.x.
- Documented capabilities: visual agent / tool invocation, fine-grained
  recognition, long context (native 256K). Useful for attribute extraction,
  constraint parsing, and explanations.

Third-party VRAM estimators (not official; treat as approximate):

| Mode | Estimate | Source |
| --- | --- | --- |
| FP16 inference | ~4.6–6.7 GB including overhead | Spheron / ANVEAI summaries |
| LoRA FP16 | ~7 GB (NVIDIA-oriented) | Spheron GPU recommender |
| Full fine-tune FP16 | ~19 GB | Spheron GPU recommender |

Official Qwen training README (mintlify / `qwen-vl-finetune`) recommends:

```
torch==2.6.0
transformers==4.57.0.dev0
deepspeed==0.17.1
flash_attn==2.7.4.post1
peft==0.17.1
```

and LoRA targeting `q_proj,k_proj,v_proj,o_proj` after freezing the base.
Flags: `tune_mm_llm`, `tune_mm_mlp`, `tune_mm_vision` (all default **False**).

### What this Mac can and cannot do

| Plan | Feasible? | Notes |
| --- | --- | --- |
| Qwen3-VL-2B HF inference, `dtype=float16`, batch 1, short decode | **Maybe** | 4.26 GB weights + activations + KV on 18 GB unified; close other apps; no FlashAttention on MPS |
| Same, multi-image or long video | **Unlikely** | Card itself recommends FA2 for multi-image/video memory |
| Official `qwen-vl-finetune` LoRA | **No** | `flash_attn` and DeepSpeed are NVIDIA-centric |
| PEFT LoRA on language projections only, tiny fashion captions | **Marginal** | Only after inference is stable; rank 8; freeze vision + merger |
| LoRA or FT on the **vision encoder** | **No** | Memory and no official MPS recipe. Do not claim it. |
| GGUF + llama.cpp Metal | **Fallback** | Official GGUF repo lists Apple Silicon / Metal |

**Trainable-module rule:** any Phase 3 run must print a table of modules with
`requires_grad=True`. If only language LoRA tensors are trainable, the report
will say “language-layer adapters only.”

## Fusion module (Phase 2)

A CLIP Combiner (2–3 linear layers, hidden 512–1024, optional attribute MLP)
is a few million parameters. Training on MPS with frozen CLIP embeddings
precomputed to disk is the **intended** training setup: the heavy encoder
runs once; the optimizer sees only the small module.

This is the only training experiment that is clearly in-scope on 18 GB.

## Storage budget (if Phase 1–3 proceed)

| Item | Approx. bytes | When |
| --- | --- | --- |
| CLIP-B/32 | 605,247,071 B | Phase 1 |
| FashionIQ images (community rar) | 988,840,917 B compressed; unzipped **unverified** | Phase 1 |
| Qwen3-VL-2B | 4,255,140,312 B | Phase 3 (deferred; over 5 GB cap) |
| Fashionpedia train images | 3,344,364,592 B | Deferred |
| Fashionpedia val/test images | 236,499,034 B | Optional Phase 4 |
| Free now | **32 Gi** | Tight if caches grow |

Do not download Qwen or Fashionpedia until Phase 1 embeddings and eval
artifacts exist and disk is re-checked.
