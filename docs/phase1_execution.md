# Phase 1 B0 execution

GitHub is the code source of truth. **Kaggle is the safer place to run B0**
on this Mac. Hugging Face Hub publishing is implemented but **disabled**.
Nothing in this document authorizes a git push, a paid GPU, Qwen, SigLIP2,
or an image re-upload.

## Safer path: Kaggle (recommended)

Local Data volume has on the order of **30 Gi** free and a **25 Gi** floor.
A local B0 would add PyTorch (~1–2 GB), CLIP (605,247,071 B), Hugging Face
temp (~40% extra), and FashionIQ images (unzipped size **unverified**).
That can exceed the **5 GB experiment cap** and can breach the disk floor.

Kaggle free notebooks have ~20 GB working disk and attach datasets without
copying them onto this Mac. Use **GPU T4**, not P100: Kaggle’s current
PyTorch CUDA wheels may lack P100 (`sm_60`) kernels.

Paid accelerators are out of scope.

## Disk policy (Mac vs Kaggle)

| Runtime | Cache location | Free-space rule |
| --- | --- | --- |
| macOS | `HF_HOME` / `~/.cache/huggingface` | Keep **25 GiB** free. A CLIP download must still leave that floor after peak+temp. |
| Kaggle | `/kaggle/working/hf-cache` (or `HF_HOME`) | Measure **actual** free bytes on the cache mount. Require CLIP weights (605,247,071 B) + 40% Hugging Face temp + 512 MiB artifact headroom. **Do not apply the Mac 25 GiB floor.** |

`styleweave.preflight.check_disk_budget` raises `PreflightError` when the
rule fails. The Kaggle notebook converts that into a cell failure and does
not write `metrics.json`.

## Coverage policy (initial B0)

Preflight fails (and refuses `metrics.json`) when any of the following is true
on the official **val** split for dress, shirt, and toptee:

- a reference image file is missing
- a target image file is missing
- a target ID is not in that category’s val gallery split
- the val gallery is incomplete

An incomplete gallery is allowed only with a **documented research
justification** in `incomplete_gallery_justification` plus
`require_complete_gallery: false`. Initial B0 has no such justification;
YAML keeps `require_complete_gallery: true`.

A failed or in-progress run **quarantines** any previous `metrics.json` to
`metrics.json.invalidated-<timestamp>` and writes `run_status.json`. Stale
numbers cannot stand in for the new run.

## What B0 computes

Frozen `openai/clip-vit-base-patch32` at revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.

```
q = normalize(encode_image(I_ref)) + normalize(encode_text(T_mod))
q = normalize(q)
```

`T_mod` is CLIP4Cir’s deterministic concat of the two relative captions:
`"{cap0.strip('.?, ').capitalize()} and {cap1.strip('.?, ')}"`.

Gallery = official per-category **val** split. Metric = Recall@10 / @50
(single target). Mean = unweighted mean of dress, shirt, toptee.

Before the full val loop, B0 encodes one real reference image and checks
that L2-sum ranks it at 1. That smoke test uses the same frozen CLIP APIs
(`get_image_features` / `get_text_features`).

Literature orientation (not our score): CLIP4Cir frozen-sum 19.70 / 38.26.
That paper’s CLIP backbone may differ; do not treat 19.70 as a gate.

## Kaggle dataset license check (do not transfer images)

| Dataset | URL | Contents | Kaggle license | Use? |
| --- | --- | --- | --- | --- |
| `nguyenph/fashion-iq` | https://www.kaggle.com/datasets/nguyenph/fashion-iq | Captions/splits only, 7.36 MB | **Unknown** | Text-only; not an image source |
| `hoangkim14/fashion-iq-dataset` | https://www.kaggle.com/datasets/hoangkim14/fashion-iq-dataset | `fashionIQ_dataset/{captions,image_splits,images}`, 74.4k files, 682.91 MB | **Other (specified in description)**; description is empty | Attach as *existing* input only. Not a verified right to **re-upload** |
| Official GitHub text | https://github.com/XiaoxiaoGuo/fashion-iq | `captions/` + `image_splits/` | CDLA, variant unspecified ([issue #22](https://github.com/XiaoxiaoGuo/fashion-iq/issues/22)) | Preferred text source |

**Do not** create a new Kaggle or Hugging Face dataset of FashionIQ
images. CDLA Permissive vs Sharing is unresolved. Coverage validation
must pass on whatever images you attach.

## Execute on Kaggle (after the hardened commit is on GitHub)

The notebook clones **GitHub `main`**. Uncommitted local files are not used.
Do not start a Kaggle run until the corrected commit is pushed.

1. Confirm the hardened commit is on
   `https://github.com/SumalyaPatnala/styleweave-vlm-lab` (`main`).
2. Open [notebooks/kaggle_phase1_b0.ipynb](../notebooks/kaggle_phase1_b0.ipynb)
   on Kaggle (**File → Import Notebook**), or paste the cells.
3. Settings: Internet **on**, accelerator **GPU T4** (not P100), free quota.
4. Add input **only** `hoangkim14/fashion-iq-dataset`.
   **Do not upload FashionIQ images.**
5. If the GitHub repo is private, add a Kaggle Secret named `GITHUB_TOKEN`
   with `repo` read scope. The notebook clones
   `https://github.com/SumalyaPatnala/styleweave-vlm-lab`.
6. Run cells **top to bottom**. Stop if any cell raises. Meaning:
   - clone repo
   - pin `transformers>=4.41,<5.3` (do not reinstall torch)
   - print Python / PyTorch / CUDA / Transformers / Pillow; require T4 and CLIP APIs
   - measure free space on `/kaggle/working/hf-cache`; require CLIP peak+temp, not 25 GiB
   - discover FashionIQ layout under `/kaggle/input`
   - dry-run coverage (missing refs/targets/gallery membership fail here)
   - smoke test on one real image, then full val encode; write
     `/kaggle/working/artifacts/phase1_frozen_clip/metrics.json` **only** on success
7. Download `metrics.json`, `report.md`, `coverage.json`, and `run_status.json`
   from `/kaggle/working/artifacts/phase1_frozen_clip/`. Copy them into this
   repo’s `artifacts/phase1_frozen_clip/` locally if you want them on disk.
   They stay gitignored until you choose to commit.

The notebook calls `styleweave.eval.b0.run_b0`. It does not copy the
ranking math. Hugging Face publishing stays off. No Qwen. No SigLIP2.

## Local MPS / CPU fallback (only if you accept the disk risk)

Repo venv: `/Users/sumalyapatnala/styleweave-vlm-lab/.venv` (Python 3.12.7).

```bash
cd /Users/sumalyapatnala/styleweave-vlm-lab
source .venv/bin/activate
pip install -e ".[phase1]"          # torch + transformers; skip if using Kaggle
python scripts/fetch_fashioniq_text.py   # ~7 MB official JSON only
df -h /System/Volumes/Data          # abort if Avail would fall under 25 Gi
PYTHONPATH=src python -m styleweave.cli --dry-run \
  --images-source "local-unverified-mirror"
```

`--dry-run` never writes `metrics.json`. Full encode:

```bash
PYTHONPATH=src python -m styleweave.cli --allow-download \
  --images-dir data/raw/fashion-iq/images \
  --images-source "document-the-mirror-url-here"
```

`--allow-download` is required for CLIP and still respects the 25 Gi floor.
Do not pass `--publish-hub`.

## Hugging Face Hub (disabled)

Publishing needs **all three**:

1. `hub.enabled: true` in the YAML (currently `false`)
2. `--publish-hub` on the CLI
3. environment `STYLEWEAVE_HF_PUBLISH=1`

Even then, image files are refused. B0 has no trainable adapter; a later
approved publish would upload `metrics.json`, `report.md`, `coverage.json`,
and [docs/model_cards/phase1_b0.md](model_cards/phase1_b0.md) only.

## Tests (no CLIP, no images from the web)

```bash
cd /Users/sumalyapatnala/styleweave-vlm-lab
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## Local preflight snapshot

Recorded from this Mac before the first Kaggle run. This is **not** a
FashionIQ result and **not** `metrics.json`.

Command:

```bash
cd /Users/sumalyapatnala/styleweave-vlm-lab
PYTHONPATH=src .venv/bin/python - <<'PY'
from styleweave.preflight import collect_environment, detect_runtime, check_disk_budget, PreflightError
print("runtime", detect_runtime())
print(collect_environment())
try:
    print(check_disk_budget(allow_download=False))
except PreflightError as exc:
    print("disk:", exc)
PY
```

Recorded 2026-10-18 on this Mac (repo `.venv`, no CLIP download):

| Check | Result |
| --- | --- |
| Unit tests | `37` passed (`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`) |
| Runtime | `macos` |
| Python | `3.12.7` |
| PyTorch / CUDA / GPU | not installed locally (`torch=null`, `cuda_available=false`) |
| Transformers / Pillow | not installed locally |
| CLIP APIs | not probeable until Transformers is installed (Kaggle pins `>=4.41,<5.3`) |
| Cache | `/Users/sumalyapatnala/.cache/huggingface` (CLIP weights **not** cached) |
| Free disk | **28.398 GiB** (`df` Avail **28Gi** on `/System/Volumes/Data`) |
| Mac 25 GiB floor | **pass** (`allow_download=false`) |
| CLIP peak+temp+headroom | 605,247,071 B × 1.40 + 512 MiB = **1,384,216,811 B** (~1.29 GiB); used on Kaggle, not as the Mac floor |
| `check_environment()` | **fails locally** (missing torch, transformers, Pillow, CLIP APIs). This is expected; Kaggle must pass it. |
| `metrics.json` | **not present** (no fabricated result) |
