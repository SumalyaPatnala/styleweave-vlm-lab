# Phase 1 B0 execution

GitHub is the code source of truth. **Kaggle is the safer place to run B0**
on this Mac. Hugging Face Hub publishing is implemented but **disabled**.
Nothing in this document authorizes a git push, a paid GPU, Qwen, or an
image re-upload.

## Safer path: Kaggle (recommended)

Local Data volume has **32 Gi** free and a **25 Gi** floor. A local B0
would add PyTorch (~1–2 GB), CLIP (605,247,071 B), Hugging Face temp, and
FashionIQ images (unzipped size **unverified**, 2.5–3.5 GB estimate). That
can exceed the **5 GB experiment cap** and can breach the disk floor if
extract is larger than estimated.

Kaggle free notebooks have ~20 GB working disk and attach datasets without
copying them onto this Mac. Use **GPU T4**, not P100: Kaggle’s current
PyTorch CUDA wheels may lack P100 (`sm_60`) kernels.

Paid accelerators are out of scope.

## Kaggle dataset license check (do not transfer images)

| Dataset | URL | Contents | Kaggle license | Use? |
| --- | --- | --- | --- | --- |
| `nguyenph/fashion-iq` | https://www.kaggle.com/datasets/nguyenph/fashion-iq | Captions/splits only, 7.36 MB | **Unknown** | Text-only; not an image source |
| `hoangkim14/fashion-iq-dataset` | https://www.kaggle.com/datasets/hoangkim14/fashion-iq-dataset | `fashionIQ_dataset/{captions,image_splits,images}`, 74.4k files, 682.91 MB | **Other (specified in description)**; description is empty | Attach as *existing* input only. Not a verified right to **re-upload** |
| Official GitHub text | https://github.com/XiaoxiaoGuo/fashion-iq | `captions/` + `image_splits/` | CDLA, variant unspecified ([issue #22](https://github.com/XiaoxiaoGuo/fashion-iq/issues/22)) | Preferred text source |

**Do not** create a new Kaggle or Hugging Face dataset of FashionIQ
images. CDLA Permissive vs Sharing is unresolved. Coverage validation
must pass on whatever images you attach.

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

Literature orientation (not our score): CLIP4Cir frozen-sum 19.70 / 38.26.
That paper’s CLIP backbone may differ; do not treat 19.70 as a gate.

## Execute on Kaggle

1. Open [notebooks/kaggle_phase1_b0.ipynb](../notebooks/kaggle_phase1_b0.ipynb) on Kaggle (File → Import notebook), or paste the cells.
2. Settings: Internet **on**, accelerator **GPU T4**.
3. Add input `hoangkim14/fashion-iq-dataset` (or another layout that already exists). **Do not upload images.**
4. If this GitHub repo is private, add a Kaggle Secret named `GITHUB_TOKEN` with `repo` read scope. The notebook clones `https://github.com/SumalyaPatnala/styleweave-vlm-lab`.
5. Run cells top to bottom. Cell 3 is a **dry-run** (coverage only). Cell 4 encodes CLIP and writes `/kaggle/working/artifacts/phase1_frozen_clip/metrics.json` only if coverage passes.
6. Download `metrics.json` and `report.md` from `/kaggle/working`. Copy them into this repo’s `artifacts/phase1_frozen_clip/` locally if you want them on disk. They stay gitignored until you choose to commit.

The notebook calls `styleweave.eval.b0.run_b0`. It does not copy the
ranking math.

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
