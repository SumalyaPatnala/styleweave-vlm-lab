# Phase 0 — Datasets and licenses

No dataset files are stored in this repository. Downloads are deferred until
Phase 1 and require enough free disk.

## FashionIQ (primary)

| Item | Official statement |
| --- | --- |
| Paper | Wu, Gao, Guo, Al-Halah, Rennie, Grauman, Feris. *Fashion IQ: A New Dataset Towards Retrieving Images by Natural Language Feedback*. CVPR 2021. https://arxiv.org/abs/1905.12794 |
| Repo | https://github.com/XiaoxiaoGuo/fashion-iq |
| License on official README | “Community Data License Agreement (CDLA) License” |
| CDLA variant | **Not specified** (Permissive 1.0 vs Sharing 1.0). Treat as research-only until clarified. |
| Intended use | Public research benchmark; supplement says “free for academic use” |
| Image origin | Product images sourced from a product-review dataset; metadata scraped from product pages |

### Official split statistics (CVPR 2021 supplement Table 6)

| Category | Train images | Val images | Test images | Train captions | Val captions | Test captions |
| --- | --- | --- | --- | --- | --- | --- |
| Dresses | 11,452 | 3,817 | 3,818 | 11,970 | 4,034 | 4,048 |
| Shirts | 19,036 | 6,346 | 6,346 | 11,976 | 4,076 | 4,078 |
| Tops & tees | 16,121 | 5,374 | 5,374 | 12,054 | 3,924 | 4,112 |

Category image counts **overlap**; unique images commonly reported in later
codebases are **74,381** (not the 77,684 category-sum). See
[SEARLE#11](https://github.com/miccunifi/SEARLE/issues/11).

Test-set target labels are **not public**. Standard CIR practice is to
report **validation** Recall@10 / Recall@50, per category and mean.

### What to download

| Artifact | Source | Size (audit) | Notes |
| --- | --- | --- | --- |
| Captions + image splits (JSON) | Official GitHub `json/` and `image_splits/` | Small (megabytes) | Prefer this over Kaggle copies |
| Attribute features | Official README “image attribute features” link | Not re-fetched | Optional for Phase 2 attribute stream |
| Images | Official README image link has been **taken down** repeatedly | — | Community mirrors exist |

### Image mirrors (not official; document provenance)

| Mirror | URL | Size | Risk |
| --- | --- | --- | --- |
| Hugging Face `Plachta/FashionIQ` | https://huggingface.co/datasets/Plachta/FashionIQ | **989 MB** `fashionIQ_dataset.rar` | Unofficial rehost; no dataset card |
| CosMo.pytorch Google Drive (historical) | https://github.com/postBG/CosMo.pytorch | Unverified | Link marked temporary |
| Yandex disk linked from official issue #18 | https://github.com/XiaoxiaoGuo/fashion-iq/issues/18 | Unverified | Reported incomplete / broken at times |
| Kaggle `nguyenph/fashion-iq` | captions/splits only in the listing we saw | 7.36 MB listing | Not a substitute for images |

**Phase 1 rule:** download official JSONs first; then one documented image
mirror; verify image IDs against official splits; log missing IDs; never
silently drop queries. If a non-trivial fraction of val images is missing,
stop and report rather than inventing a private split.

## Fashionpedia (optional)

| Item | Official statement |
| --- | --- |
| Paper | Jia et al., *Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset*, CVPR 2020 |
| Project | https://fashionpedia.github.io/home/ |
| Download | https://github.com/cvdfoundation/fashionpedia |
| Annotations / ontology / website | **CC BY 4.0** |
| Images | Fashionpedia **does not own** image copyright. Use must follow Flickr, Unsplash, Burst, Freestocks, Kaboompics, Pexels terms. |
| Train images zip | https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip — **3,344,364,592 bytes (~3.34 GB)** (HTTP HEAD 2026-10-18) |
| Val/test images zip | https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip — **236,499,034 bytes (~226 MB)** |
| Eval | Fashionpedia API `FPEval` (IoU + attribute F1). https://github.com/KMnP/fashionpedia-api |

**Decision:** do not download Fashionpedia images in Phase 1. If Phase 4 needs
attribute localization, start with **val images (226 MB) + JSON annotations**.

## Pretrained models

| Model | Card | License | On-disk weights | Hardware note |
| --- | --- | --- | --- | --- |
| `openai/clip-vit-base-patch32` | https://huggingface.co/openai/clip-vit-base-patch32 | OpenAI CLIP **code** is MIT; HF card has deployment restrictions / safety caveats. Weights commonly treated as research-usable; confirm card before any product use. | ~605 MB (`pytorch_model.bin`) | Inference ~0.3–0.6 GB |
| `Qwen/Qwen3-VL-2B-Instruct` | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct | **Apache-2.0** (model card YAML, 2026-10-18) | **4.26 GB** `model.safetensors` | Official docs recommend `transformers>=4.57` and FlashAttention-2 (CUDA) |
| `Qwen/Qwen3-VL-2B-Instruct-GGUF` | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF | Same family | Quantized LLM + mmproj | llama.cpp / Metal path if HF+MPS OOMs |

CLIP card (OpenAI): deployed unconstrained use is **out of scope**; research
image search in a constrained, tested taxonomy is the intended setting. This
lab stays in that setting.

## Code license of this repo

MIT for original StyleWeave code. Third-party data and weights stay with
their owners. See `NOTICE`.
