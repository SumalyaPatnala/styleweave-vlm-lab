---
license: mit
library_name: transformers
tags:
  - composed-image-retrieval
  - fashion-iq
  - clip
  - research-only
---

# StyleWeave Phase 1 B0 (frozen CLIP L2-sum)

Research evaluation of frozen `openai/clip-vit-base-patch32` on the official
FashionIQ **validation** split. This is not a new architecture and not a
claim of state of the art.

- Query: `normalize(image) + normalize(text)`, then L2 (Phase 0 B0).
- No trainable adapter is produced in B0.
- FashionIQ images are **not** redistributed here.
- CLIP unconstrained deployment remains out of scope (OpenAI CLIP card).
- Metrics in `metrics.json` are valid only if `produced_by` is
  `styleweave.eval.b0`.

Do not upload this card until Hugging Face publishing is explicitly
approved (`STYLEWEAVE_HF_PUBLISH=1` and `--publish-hub`).
