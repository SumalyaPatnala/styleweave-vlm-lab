# Dataset audit

No dataset images are stored in this repository. Phase 0 does **not**
download archives. Byte sizes below were re-checked with HTTP HEAD or the
GitHub / Hugging Face APIs on **2026-10-18**. Bodies were not fetched.

## FashionIQ (primary, Track A / C)

| Field | Value |
| --- | --- |
| Paper | Wu, Gao, Guo, Al-Halah, Rennie, Grauman, Feris. CVPR 2021 |
| URL | https://arxiv.org/abs/1905.12794 |
| Official repo | https://github.com/XiaoxiaoGuo/fashion-iq |
| License on official README | “Community Data License Agreement (CDLA) License” |
| CDLA variant | **Not specified.** Open question [issue #22](https://github.com/XiaoxiaoGuo/fashion-iq/issues/22) (2025-02-06, unanswered as of 2026-10-18). Treat as **academic research use only**. Do not redistribute images. Prefer CDLA-Permissive vs Sharing clarification before any public data dump. |
| Image origin | Product catalog images; metadata from product pages |
| Unique images (community consensus) | 74,381; category counts overlap so the paper’s per-category sums are larger ([SEARLE #11](https://github.com/miccunifi/SEARLE/issues/11)) |
| Report split | **Validation** (test targets unpublished) |

Official supplement counts (CVPR 2021 appendix):

| Category | Train img | Val img | Test img | Val captions |
| --- | --- | --- | --- | --- |
| Dresses | 11,452 | 3,817 | 3,818 | 4,034 |
| Shirts | 19,036 | 6,346 | 6,346 | 4,076 |
| Tops & tees | 16,121 | 5,374 | 5,374 | 3,924 |

### Official text artifacts (GitHub, not `json/`)

The official tree uses `captions/` and `image_splits/`, **not** a `json/`
directory. Listing retrieved from the GitHub Contents API on 2026-10-18.

| Path | Bytes |
| --- | --- |
| `captions/cap.{dress,shirt,toptee}.{train,val,test}.json` (9 files) | 5,748,255 |
| `image_splits/split.{dress,shirt,toptee}.{train,val,test}.json` (9 files) | 1,398,312 |
| **Official text total** | **7,146,567 (~6.8 MiB)** |

Val-only text (Phase 1 B0): three `cap.*.val.json` + three
`split.*.val.json` ≈ 1.47 MB.

Test caption JSON files exist but **do not include target labels**. Do
not treat them as an evaluation split.

### Download plan (not executed)

| Artifact | Source | Size (verified) | Phase 0 | Peak disk if downloaded later |
| --- | --- | --- | --- | --- |
| Captions + splits | Official GitHub `captions/`, `image_splits/` | 7,146,567 B | Allowed later as text-only | <20 MB with git clone of that repo |
| Attribute features | Official README link | Not re-fetched (link historically taken down) | No | Estimate before fetch |
| Images (official) | Official README “images can be downloaded from here” | Link repeatedly **taken down** | No | — |
| Community rar `Plachta/FashionIQ` | https://huggingface.co/datasets/Plachta/FashionIQ | **988,840,917 B** (`fashionIQ_dataset.rar`; HF `x-linked-size` 2026-10-18). No dataset card; only that rar. | **No** (Phase 0) | rar + extract + temp ≈ **2.5–3.5 GB** (**estimate**; unzipped size is **unverified** until a future `du`) |

**Risks that remain unverified until Phase 1:**

1. Unofficial mirrors may be incomplete versus official split IDs.
2. Unzipped image footprint is unknown; abort if extract would leave
   `< 25 Gi` free.
3. Some images appear in more than one category.

**Phase 1 rule:** download official captions/splits first; then one
documented image mirror; verify every official **val** ID; fail the run
on any missing **target**; fail if `>1%` of the gallery is missing; log
the missing-ID list. Do not invent a private split.

Kaggle copies that contain only captions/splits are not image sources.

## Fashionpedia (optional, Track C attributes)

| Field | Value |
| --- | --- |
| Paper | Jia et al., CVPR 2020. https://arxiv.org/abs/2004.12276 |
| Project | https://fashionpedia.github.io/home/ |
| Host | https://github.com/cvdfoundation/fashionpedia |
| Annotations / ontology / website | **CC BY 4.0** ([data license page](https://fashionpedia.github.io/home/data_license.html), re-checked 2026-10-18) |
| Images | Fashionpedia **does not own** copyright. Flickr, Unsplash, Burst, Freestocks, Kaboompics, Pexels terms apply. |
| Train images zip | https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip — HTTP HEAD **3,344,364,592 B** (2026-10-18) |
| Val/test images zip | https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip — HTTP HEAD **236,499,034 B** (2026-10-18) |
| Eval | https://github.com/KMnP/fashionpedia-api (`FPEval`, IoU + attribute F1) |

**Decision:** do not download Fashionpedia in Phase 0 or Phase 1. If
attribute localization is needed later, start with **val/test (226 MB)**
only. Train images (3.34 GB) would consume most of the 5 GB experiment
budget.

Fashionpedia photos are street / celebrity images, not FashionIQ catalog
shots. Domain shift must be stated.

## Counterfactual subset (to be built in Phase 4, not now)

`styleweave-cf-val`: 100–200 FashionIQ **validation** queries with
human-checked preserve / change / avoid slots. Never used for training.
Not a published dataset yet; construction protocol is in
`docs/experiment_plan.md`.

## Licensing summary for this lab

| Asset | License | Redistribute in git? |
| --- | --- | --- |
| StyleWeave code | MIT | Yes |
| FashionIQ captions / splits | CDLA (variant unclear; issue #22 open) | Prefer download script, not a vendored dump |
| FashionIQ images | Same dataset; unofficial mirrors | **Never commit** |
| Fashionpedia annotations | CC BY 4.0 | Optional later, with attribution |
| Fashionpedia images | Source-site terms | **Never commit** |
| CLIP weights | Research card; unconstrained deployment out of scope | **Never commit** |
| Qwen3-VL-2B weights | Apache-2.0 | **Never commit** |
