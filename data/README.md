# Data directory

Raw datasets and images are **not** stored in git. Place downloads under
`raw/` after reading `docs/dataset_audit.md`.

Expected later layout (not created until Phase 1):

```
data/raw/fashion-iq/captions/          # official cap.{dress,shirt,toptee}.{train,val,test}.json
data/raw/fashion-iq/image_splits/      # official split.{dress,shirt,toptee}.{train,val,test}.json
data/raw/fashion-iq/images/
data/cache/clip_embeddings/
```

There is no official `json/` directory. Use `captions/` + `image_splits/`.

Fetch official text only:

```
python scripts/fetch_fashioniq_text.py
```

Do not commit images. Do not upload FashionIQ images to Kaggle or Hugging Face.
On Kaggle, attach an existing dataset and pass `--data-root` / `--images-source`.
See `docs/phase1_execution.md`.
