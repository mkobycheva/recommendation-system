# MVP app (SVD + ALS + item2vec + SASRec + BERT4Rec, Gradio)

FastAPI backend + Gradio frontend for cold-start cross-domain recommendations
(books + movies). See [../notebooks/00_export_metadata.ipynb](../notebooks/00_export_metadata.ipynb)
for how `items_metadata.parquet` is produced in Colab, and the export cells at
the end of the `svd`/`als`/`item-2-vec`/`transformer` branch notebooks for how
each model's artifacts are produced:
`item_factors.npy`/`item2idx.json`/`domain_item_indices.json` (svd, als),
`word2vec.model`/`domain_item_indices.json` (item2vec), or
`model_state_dict.pt`/`item2idx.json`/`domain_item_indices.json`/`config.json`
(sasrec, bert4rec). SASRec/BERT4Rec cold-start treats the cart as a
pseudo-sequence in selection order -- there's no real timestamp for a new
user's cart, so this is an approximation, not the training-time sequence.

## Install

```bash
pip install -r app/requirements.txt
```

## Run locally on synthetic data (no real artifacts needed)

Generate a small synthetic `ARTIFACTS_DIR` (10 books + 10 movies, random
factors / a tiny trained item2vec model / untrained tiny SASRec+BERT4Rec,
toy titles):

```bash
python -m app.scripts.make_synthetic_artifacts ./artifacts
```

Start the backend:

```bash
ARTIFACTS_DIR=./artifacts uvicorn app.api.main:app --reload
```

In a separate terminal, start the frontend:

```bash
python app/frontend/app.py
```

Gradio prints a local URL (default `http://127.0.0.1:7860`). The backend must
already be running -- the frontend only talks to it over HTTP (`API_URL`, see
below), it never imports `app.recommenders` directly.

## What to change for real Colab artifacts

- `ARTIFACTS_DIR` -- point it at wherever the Colab-produced artifacts land
  (e.g. a synced copy of the shared Drive folder:
  `.../recsys-data/artifacts`). Expected layout:
  ```
  <ARTIFACTS_DIR>/
    items_metadata.parquet
    svd/item_factors.npy, item2idx.json, domain_item_indices.json
    als/item_factors.npy, item2idx.json, domain_item_indices.json
    item2vec/word2vec.model, domain_item_indices.json
    sasrec/model_state_dict.pt, item2idx.json, domain_item_indices.json, config.json
    bert4rec/model_state_dict.pt, item2idx.json, domain_item_indices.json, config.json
  ```
- `ITEMS_METADATA_PATH` -- only needed if `items_metadata.parquet` doesn't
  live directly under `ARTIFACTS_DIR` (defaults to `ARTIFACTS_DIR/items_metadata.parquet`).
- `API_URL` (frontend only) -- if the backend isn't on `http://localhost:8000`.

The backend fails fast at startup (not on the first request) if any of these
files are missing, with the exact path it looked for.

## Tests

```bash
pytest tests/test_recommenders.py tests/test_sequential_recommenders.py tests/test_api.py -v
```

Both use in-memory / `tmp_path` synthetic fixtures -- no real dataset or
Colab artifacts required.

## Known limitations of this MVP

- `/search` fuzzy match is title-only (rapidfuzz `WRatio`); no
  ranking by popularity.
- Gradio cart/search state is per-session (`gr.State`), not persisted.
- Search results and recommendations render as `gr.Gallery` with covers
  (`image_url`); items with no cover in `items_metadata.parquet` fall back to
  a placeholder image.
- SASRec/BERT4Rec cold-start uses cart selection order as a stand-in for a
  real interaction sequence -- reordering the same cart can change results.
