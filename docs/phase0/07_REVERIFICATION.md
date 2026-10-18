# Phase 0 — Re-verification log

**Date:** 2026-10-18 (second pass, after the GitHub remote existed)
**Scope:** documentation only. No weight or image bodies downloaded.
**Isolation:** WorldSense-AI and portfolio trees were not modified.

This note records what was re-checked after the first Phase 0 packet and
the GitHub push. Canonical conclusions live in `docs/*.md`.

## Working tree at re-check

- Branch: `main`, tracking `origin/main` at `5c499f6` before this commit.
- Untracked canonical docs (`docs/hardware_feasibility.md` and siblings)
  existed locally but were not yet on GitHub.
- Remote: `https://github.com/SumalyaPatnala/styleweave-vlm-lab.git`
  (`git ls-remote` succeeded). `gh` CLI was **not** logged in; git used
  the osxkeychain helper.

## Environment (measured)

| Item | Value |
| --- | --- |
| Chip / memory | Apple M3 Pro, 18 GB unified, Metal 4 |
| Disk free | **32 Gi** on `/System/Volumes/Data` (was ~33–34 Gi earlier the same day) |
| Repo venv | Python 3.12.7, torch **not** installed, ~12 MB |
| HF hub cache | 67 GB; **no** CLIP / SigLIP2 / Qwen3-VL repos |
| Nested WorldSense-AI | Absent |

## Source re-checks (HEAD / API, no body download)

| Asset | Evidence | Result |
| --- | --- | --- |
| FashionIQ README license | GitHub README | “CDLA License”; variant unspecified |
| CDLA variant | [issue #22](https://github.com/XiaoxiaoGuo/fashion-iq/issues/22) | Still open / unanswered |
| Official text layout | GitHub Contents API | `captions/` + `image_splits/` (not `json/`); **7,146,567 B** |
| Unique image count | [SEARLE #11](https://github.com/miccunifi/SEARLE/issues/11) | 74,381; report on val |
| Plachta image rar | HF `x-linked-size` | **988,840,917 B**; no dataset card |
| Fashionpedia train zip | S3 HEAD | **3,344,364,592 B** |
| Fashionpedia val/test zip | S3 HEAD | **236,499,034 B** |
| Fashionpedia license page | project site | CC BY 4.0 annotations; images remain source-site terms |
| CLIP-B/32 weights | HF `x-linked-size` | **605,247,071 B** |
| CLIP card use limits | HF README | Unconstrained deployment out of scope |
| SigLIP2-base-224 | HF `x-linked-size` | **1,500,800,904 B**; Apache-2.0 |
| Qwen3-VL-2B Instruct | HF `x-linked-size` + card YAML | **4,255,140,312 B**; 2.13B BF16; Apache-2.0 |
| WISER | arXiv:2602.23029, CVPR 2026 OA, GitHub | Exists; authors Wang et al. |
| PinPoint | arXiv:2603.04598, CVPR 2026 OA | Exists; authors Mahadev et al. |
| CLIP4Cir Table 1 | arXiv:2308.11485 | Frozen sum 19.70/38.26; frozen Combiner 30.96/53.56; FT+Combiner 40.80/64.20 |

## Still unverified (do not treat as measured)

1. Unzipped FashionIQ image disk after extracting the Plachta rar.
2. ID coverage of that rar versus official val split files.
3. CLIP process RSS / MPS encode speed on this Mac.
4. Qwen3-VL-2B peak RSS on MPS.
5. CDLA-Permissive vs Sharing (maintainer has not answered).
6. Official FashionIQ image host (README link still taken down).

## Corrections applied in the canonical packet

- Official FashionIQ text lives under `captions/`, not `json/`.
- Disk free updated from ~33–34 Gi to **32 Gi**.
- GitHub remote is recorded as existing; do not `gh repo create` again.
- B0 is explicitly the CLIP4Cir frozen-sum analogue; the 40.80 FT row is
  out of scope.
- Byte sizes are stored as integers from HEAD/API, not rounded guesses.
