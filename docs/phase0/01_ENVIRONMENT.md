# Phase 0 — Environment

Measured on 2026-10-18 from this Mac. Hardware identifiers that are not
needed for reproducibility (serial numbers, UDIDs) are omitted.

## Isolation

| Path | Role | Action taken |
| --- | --- | --- |
| `/Users/sumalyapatnala/styleweave-vlm-lab` | This research repo | Created |
| `/Users/sumalyapatnala/WorldSense-AI` | Separate project | Inspected only (`git status`); not modified |
| `/Users/sumalyapatnala/PycharmProjects/Sumalya Patnala Portfolio` | Portfolio | Listed only; not modified |
| `/Users/sumalyapatnala/PycharmProjects/MyPortfolio` | Older portfolio | Listed only; not modified |

The new repository is not nested inside any of those trees.

## Hardware

| Item | Value |
| --- | --- |
| Model | MacBook Pro, Mac15,6 |
| Chip | Apple M3 Pro |
| CPU | 11 cores (5 performance + 6 efficiency) |
| GPU | 14-core Apple GPU, Metal 4 |
| Memory | 18 GB unified (19327352832 bytes) |
| OS | Darwin 25.6.0 (arm64) |
| CUDA | None |
| MPS | Expected once PyTorch is installed (Apple Silicon + Metal 4) |

Memory pressure at audit time was already non-trivial (large historical
swap-in / swap-out counts; compressor in use). Treat 18 GB as **shared with
the OS and browser**, not as dedicated VRAM.

## Storage

| Mount | Size | Used | Free | Notes |
| --- | --- | --- | --- | --- |
| `/System/Volumes/Data` | 460 Gi | 383 Gi | **33 Gi** | 93% full |

Large existing caches (not deleted; approval required):

| Path | Size |
| --- | --- |
| `~/.cache/huggingface` | 67 G |
| `~/Library/Caches` | 25 G |
| `~/.cache/torch` | 335 M |

Cached Hugging Face hubs include Mistral-7B, GPT-Neo-2.7B, Stable Diffusion,
Phi-3, Whisper-large, and others. **No CLIP and no Qwen3-VL weights** were
present. Reusing unused caches is optional and was not done.

## Python

| Interpreter | Version | Path |
| --- | --- | --- |
| System / Homebrew default `python3` | 3.14.6 | `/opt/homebrew/opt/python@3.14/bin/python3.14` |
| Homebrew 3.12 | **3.12.7** | `/opt/homebrew/opt/python@3.12/bin/python3.12` |
| Project venv | 3.12.7 | `/Users/sumalyapatnala/styleweave-vlm-lab/.venv/bin/python` |

PyTorch was **not** installed in any inspected interpreter at audit time.
Conda `base` exists at `/opt/anaconda3`; it is not used for this project.

Available on PyPI at audit time (not installed):

- `torch` 2.14.1 (and 2.6.0, the version named in official Qwen training docs)
- `transformers` 5.20.1 (Qwen3-VL first landed around 4.57)

## What was created

- Git repository at `/Users/sumalyapatnala/styleweave-vlm-lab`
- Python 3.12.7 virtualenv at `.venv` (empty of packages)
- Phase 0 documentation under `docs/phase0/`

## What was not done

- No `pip install`
- No dataset download
- No model weight download
- No training
- No git remote, push, or deploy
- No deletion of the 67 GB Hugging Face cache
