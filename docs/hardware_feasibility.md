# Hardware feasibility

**Machine (re-checked 2026-10-18):** MacBook Pro Mac15,6, Apple M3 Pro,
18 GB unified memory (19,327,352,832 bytes), 11 CPU cores (5P+6E),
14-core GPU, Metal 4, Darwin 25.6.0 arm64 (macOS 26.6.2).
**Disk at last check:** **32 Gi** free on `/System/Volumes/Data`
(460 Gi container, 93% full). Earlier the same day this was ~33–34 Gi.
**Acceleration:** PyTorch MPS after a future install. No CUDA.
**Policy:** reserve ≥25 GB free. New Phase 0/1 artifacts ≤5 GB including
temporary files. No large downloads in Phase 0.

## What this Mac can run

| Work | Feasible now? | Memory | Disk (download + extract + temp) | Notes |
| --- | --- | --- | --- | --- |
| Frozen CLIP ViT-B/32 inference | Yes | ~0.6–1.5 GB process | 605,247,071 B weights + HF temp | Default Track A encoder |
| Precompute FashionIQ val embeddings | Yes | ~2 GB peak | Embeddings ≪1 GB if float16 | Preferred before any training |
| Small Combiner + InfoNCE on cached embeddings | Yes | <4 GB | Checkpoints of a few MB | Only training that is clearly in-scope |
| SigLIP2-base-patch16-224 frozen | Maybe later | ~1–2 GB | **1,500,800,904 B** weights | Heavier than CLIP; not first baseline |
| Qwen3-VL-2B Instruct inference, batch 1 | Conditional | 4,255,140,312 B weights + KV; 18 GB shared | **4.26 GB** + HF temp | No FlashAttention on MPS; RSS **unmeasured** |
| Official Qwen LoRA (`flash_attn`, DeepSpeed) | No | N/A | N/A | NVIDIA-oriented recipe |
| Language-only PEFT, tiny split | Marginal | Often ≥7 GB in NVIDIA estimates | Adapter MB + 4.26 GB base | Only after a successful dry-load |
| Vision-encoder fine-tune (CLIP or Qwen) | No | Exceeds a safe 18 GB budget | — | Do not claim it |
| Full Fashionpedia train images | Avoid | — | **3,344,364,592 B** zip + extract | Breaks the 5 GB experiment cap if combined with Qwen |

## Apple Silicon / MPS notes

- Unified memory is shared with macOS, Cursor, and the browser. Treat
  **14 GB** as a soft process ceiling, not 18 GB.
- FlashAttention-2 is CUDA-only. Official Qwen3-VL training docs recommend
  `flash_attn` and DeepSpeed
  ([Qwen3-VL training overview](https://www.mintlify.com/QwenLM/Qwen3-VL/fine-tuning/overview)).
  That stack is **not portable** to this Mac.
- MPS is appropriate for CLIP encode + a tiny MLP. It is unproven here for
  Qwen3-VL-2B until a dry-load is approved.
- GGUF + llama.cpp Metal is a fallback for Track B inference if Hugging Face
  + MPS OOMs ([Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF)).

## Python environment

| Interpreter | Use? |
| --- | --- |
| Homebrew / system `python3` 3.14.6 | No |
| Existing repo `.venv` with **Python 3.12.7** | Yes — do not create a second venv |
| Conda envs under `/opt/anaconda3` or other project venvs | Do not modify |

Re-checked 2026-10-18: `.venv` is 3.12.7, **torch is not installed**,
pip package count is the empty venv bootstrap only (~12 MB).

## Isolation

This file describes `/Users/sumalyapatnala/styleweave-vlm-lab` only.
WorldSense-AI, portfolio repositories, global Hugging Face caches, and
system caches must not be deleted or rewritten.
