# Resource budget

Hard limits for this machine:

1. Keep **≥25 GB free** on the Data volume.
2. New local project artifacts + dependencies for initial experiments:
   **≤5 GB** (downloads, extracts, temps, venv growth).
3. Do not clean `~/.cache/huggingface` (67 GB) or other repos.
4. Do not download during Phase 0.

## Current footprint (re-checked 2026-10-18)

| Item | Size | Counts against 5 GB cap? |
| --- | --- | --- |
| StyleWeave working tree (git + docs) | ~1 MB tracked | Yes, negligible |
| Existing repo `.venv` (empty of ML pkgs) | ~12 MB | Yes, already created; **do not clone another venv** |
| Free disk | **32 Gi** | Must stay ≥25 Gi |
| Global HF cache | 67 GB | **No** — do not delete; also **no CLIP/Qwen** to reuse |

Phase 0 added **markdown only**. No torch, no CLIP, no FashionIQ images.

## Future download estimates (approval required)

Include **temporary** Hugging Face / unzip overhead (~20–40%). Sizes marked
“verified” are HTTP `x-linked-size` or GitHub `size` on 2026-10-18.

| Future fetch | Download (verified unless noted) | Peak disk (download + extract/temp) | Residual after cleanup of temps | 5 GB cap |
| --- | --- | --- | --- | --- |
| FashionIQ official captions + splits | 7,146,567 B | <20 MB | ~7 MB | OK |
| CLIP ViT-B/32 | 605,247,071 B | ~0.8 GB | ~0.61 GB | OK |
| FashionIQ images (`Plachta` rar) | 988,840,917 B | **2.5–3.5 GB** (unzipped size **unverified**) | unknown until measured | Tight; CLIP+JSON+images may consume most of the 5 GB |
| SigLIP2-base-224 | 1,500,800,904 B | ~1.9 GB | ~1.50 GB | Only instead of, not plus, extra VLMs |
| Qwen3-VL-2B | 4,255,140,312 B | **~5.2–5.5 GB** | 4.26 GB | **Exceeds 5 GB cap** unless you explicitly raise it |
| Fashionpedia train zip | 3,344,364,592 B | ~6–7 GB unzipped peak | 3.3 GB+ | **Over cap — deferred** |
| Fashionpedia val/test zip | 236,499,034 B | ~0.4 GB | 0.23 GB | OK later, alone |

**B0 peak sketch (not started):** CLIP ~0.8 GB + JSON <20 MB + FashionIQ
images 2.5–3.5 GB estimate ≈ **3.3–4.3 GB** new. With 32 Gi free, residual
free could land near **28 Gi** if estimates hold — still above 25 Gi, but
with little slack. Abort mid-extract if `Avail` approaches the floor.

**Rule before any fetch:** write the row above into the run ticket, run
`df -h /System/Volumes/Data`, and abort if `Avail - peak < 25 Gi`.

## Memory budget (unified 18 GB)

| Job | Estimate | Soft ceiling |
| --- | --- | --- |
| Frozen CLIP encode, batch 8–32 | 1–3 GB | OK |
| Combiner training on cached embeddings | <4 GB | OK |
| Qwen3-VL-2B generate, batch 1 | 6–12 GB (uncertain on MPS) | Abort if RSS > 14 GB |
| Qwen LoRA | Often cited ~7 GB+ on NVIDIA | Not in Phase 0; likely skip |
| Full FT | ~19 GB cited | Impossible here |

## Dependency reuse

- Use the **existing** `/Users/sumalyapatnala/styleweave-vlm-lab/.venv`.
- Do not install PyTorch until Phase 1 is approved.
- Do not import or modify WorldSense-AI, portfolio, or other project venvs.
- Do not delete global caches to “make space.”

## Git / GitHub

- Local root: `/Users/sumalyapatnala/styleweave-vlm-lab`
- Remote: `https://github.com/SumalyaPatnala/styleweave-vlm-lab` (`main`).
  Do not create a second repository.
- Do not commit `.venv/`, `.env`, datasets, `*.safetensors`, checkpoints.
- Do not push to WorldSense-AI or any portfolio remote.
