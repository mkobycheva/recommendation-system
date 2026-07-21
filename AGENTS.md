# AGENTS.md - Recommender System Project

## What this project is

A cross-domain recommender system for books and movies using the Amazon
Reviews 2023 dataset. The core research question: can a model trained on a user's
ratings in one domain reliably predict their preferences in another?

The current project scope is limited to two domains only: Books and Movies & TV.
Do not add a third domain for modeling, preprocessing, evaluation, or notebook work
unless the project scope is explicitly changed later.

The approach is to establish baselines with simple models, then progressively build
toward sequential and ensemble methods, evaluating each on cross-domain transfer.

## Current status

SVD, ALS, item2vec, SASRec, and BERT4Rec are all trained (each on its own
branch: `svd`, `als`, `item-2-vec`, `transformer`) and evaluated with shared
NDCG@10/Recall@10/MAP@5 metrics plus Top-Popular/Random baselines. Ensemble
(`04_ensemble.ipynb`) has not been done yet.

On top of that, this `app` branch adds an MVP web app (FastAPI + Gradio) that
serves cold-start cross-domain recommendations from all five trained models,
and it is deployed publicly on Google Cloud Run. See "MVP web app &
deployment" below before touching `app/`, `space/`, `Dockerfile`, or
`DEPLOY.md`.

---

## Project structure

```text
recommender-system/
├── data/
│   ├── raw/              <- never modify; gitignored
│   ├── processed/
│   └── splits/
├── notebooks/
│   ├── 00_eda.ipynb            <- exploratory data analysis
│   ├── 00_export_metadata.ipynb  <- exports items_metadata.parquet + publishes
│   │                               all model artifacts to a public HF Dataset repo
│   ├── 01_baselines.ipynb      <- SVD and ALS (see also 01_1_svd.ipynb)
│   ├── 02_item2vec.ipynb
│   ├── 03_sequential.ipynb     <- LSTM, SASRec, BERT4Rec
│   └── 04_ensemble.ipynb       <- not done yet
├── src/
│   ├── data/
│   │   ├── overlap_check.py   <- run this first
│   │   ├── preprocess.py
│   │   ├── build_matrix.py
│   │   └── train_test_split.py
│   └── evaluation/
│       ├── metrics.py         <- NDCG@K, Recall@K, Precision@K, RMSE
│       └── cross_domain_eval.py
├── configs/
│   └── model_configs.py       <- all hyperparameters as Python dicts
├── results/
├── app/                        <- MVP web app (FastAPI + Gradio), see app/README.md
│   ├── recommenders/           <- cold-start inference wrappers, one per model
│   ├── api/main.py             <- FastAPI backend (/search, /recommend)
│   ├── frontend/app.py         <- Gradio UI, talks to the backend over HTTP
│   └── scripts/make_synthetic_artifacts.py  <- fake artifacts for local dev
├── space/                      <- single-process Gradio entry point for deployment
│   └── main.py                 <- what Dockerfile/Cloud Run actually run
├── tests/                       <- pytest; covers src/, app/recommenders/, app/api/
├── Dockerfile, .dockerignore    <- Cloud Run image, builds space/main.py
├── DEPLOY.md                    <- gcloud run deploy instructions
├── requirements.txt              <- for notebooks (Colab); app/ has its own
└── README.md
```

---

## What goes where

**Notebooks** (`notebooks/`) are where models are built and experiments are run.
Each notebook covers one stage of the project. They are the primary deliverable.
Clear all outputs before committing: `Kernel -> Restart & Clear Output`.

**`src/`** contains reusable logic that every notebook imports - data loading,
preprocessing, matrix construction, and evaluation metrics. Write something once
here so all notebooks share the same code path. When you figure something out in a
notebook, extract the reusable parts into `src/`.

**`src/data/`** handles everything before modeling: downloading, cleaning, building
matrices, splitting. The `overlap_check.py` script must be run first - it verifies
that enough users exist across Books and Movies to make cross-domain transfer viable.

**`src/evaluation/`** contains shared metric functions used by every notebook.
Write this before training any model. Primary metrics are NDCG@10 and Recall@10.
Secondary metrics are Precision@10 and RMSE. MAP@K is implemented here
(`map_at_k` in `metrics.py`) as shared Python code, not only inside a notebook.

**`configs/model_configs.py`** stores all hyperparameters as plain Python dicts.
Never hardcode hyperparameters inside a notebook. Import them from here instead.

**`data/`** is gitignored. Never commit the Amazon dataset. Raw files go in
`data/raw/` and are never modified. Processed outputs go in `data/processed/`.
Train/val/test splits go in `data/splits/`.

**`results/`** stores metric outputs and plots produced by notebooks.

---

## MVP web app & deployment

`app/` is a working FastAPI + Gradio app, separate from the notebook/`src/`
research pipeline above. It serves cold-start cross-domain recommendations
(a cart of items the model never trained on, folded into each model's
embedding space) from all five trained models. See
[app/README.md](app/README.md) for how to run it locally against synthetic
or real artifacts.

`space/main.py` is a second, single-process entry point for deployment
(container hosts / HF Spaces expect one process, not the FastAPI+Gradio
split `app/` uses locally) — it re-imports `app/recommenders/` directly and
duplicates `app/frontend/app.py`'s UI. When changing recommendation or UI
logic, update both `app/` and `space/main.py` together; nothing enforces
they stay in sync automatically.

Model artifacts (~4GB) are not committed to git. `00_export_metadata.ipynb`
publishes them, plus `items_metadata.parquet`, to a public Hugging Face
Dataset repo; `space/main.py` downloads them at container startup via
`ARTIFACTS_REPO`/`snapshot_download`.

Deployed on Google Cloud Run — see [DEPLOY.md](DEPLOY.md) for the `gcloud`
commands, current flags (`--min-instances 0`, `--memory 8Gi`, `--cpu-boost`,
a long `--startup-probe`), and the reasoning behind each one.

---

## Current ALS baseline scope

Completed — kept here as a record of the constraints the trained `als`
branch model was built under, in case it needs to be reproduced or retrained.

For the first ALS baseline, use only the Books + Movies user intersection.

Use the prepared Google Drive split CSVs directly with `pd.read_csv`:
`books_train.csv`, `books_valid.csv`, `books_test.csv`, `movies_train.csv`,
`movies_valid.csv`, and `movies_test.csv`.

Do not implement or depend on `src/data/preprocess.py` for this ALS stage. The
prepared split files are assumed to already contain the intersecting users and
valid train/validation/test splits.

Train one collective ALS model on merged Books and Movies train interactions.
Create globally unique item IDs with a domain prefix, then filter recommendations
back to the requested target domain before evaluating top-K metrics.
MAP@K must be implemented in shared Python code under `src/evaluation/`, not only
inside the notebook.

---

## Data

Source: Amazon Reviews 2023 - https://amazon-reviews-2023.github.io

Domains used: Books, Movies & TV.

Key rules:
- Use `parent_asin` as the item identifier, not `asin`.
- Keep only: `user_id`, `parent_asin`, `rating`, `timestamp`.
- Apply 5-core filtering (users and items with at least 5 interactions).
- Split by time, not randomly: for each user, last interaction = test,
  second-to-last = validation, everything before = train.
- Never write to `data/raw/`. Only read from it.

---

## Google Colab setup

Code lives on GitHub. Data lives on a shared Google Drive folder
(`MyDrive/recsys-data/raw/`). One person uploads the data once; everyone else
accesses the same shared folder.

### Step 1 - Upload the data (one person, one time only)

Run this in a Colab cell to download the raw files directly from the Amazon dataset
into the shared Drive folder, skipping any local download:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/recsys-data/raw', exist_ok=True)
%cd /content/drive/MyDrive/recsys-data/raw

!wget -q https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz
!wget -q https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Movies_and_TV.jsonl.gz
```

Then share the `recsys-data/` folder with all teammates via Google Drive.
Books.jsonl.gz is several GB - this will take a while. Do it once, never again.

### Step 2 - Notebook setup (every person, every notebook)

Paste this at the top of every notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/your-org/recommender-system.git 2>/dev/null \
  || (cd recommender-system && git pull)

%cd recommender-system
!pip install -r requirements.txt -q

import sys, os
sys.path.insert(0, '.')

os.makedirs('data/raw', exist_ok=True)
!ln -s /content/drive/MyDrive/recsys-data/raw data/raw
```

The symlink makes `data/raw/` in the repo point to the shared Drive folder, so all
paths in `src/` work consistently without hardcoding Drive paths anywhere.

Then import shared modules normally:

```python
from src.data.preprocess import load_ratings
from src.evaluation.metrics import ndcg_at_k
```

---

## Development order

Steps 1-8 are done. Each is kept here as a record of what unblocked the
next step, in case any stage needs to be reproduced or redone.

1. **`src/data/overlap_check.py`** - load Books and Movies, find users appearing
   in both domains, report count. If overlap is below ~10k users, revisit sampling.

2. **`src/data/preprocess.py` + `build_matrix.py`** - clean ratings, build sparse
   user-item matrices per domain, apply 5-core filtering.

3. **`src/data/train_test_split.py`** - time-based leave-one-out split.

4. **`src/evaluation/metrics.py`** - implement NDCG@10, Recall@10, Precision@10,
   RMSE, and `cross_domain_eval.py` for train-on-A / test-on-B evaluation.

5. **`notebooks/00_eda.ipynb`** - explore the data: rating distributions, user
   activity, item popularity, overlap counts, sparsity per domain.

6. **`notebooks/01_baselines.ipynb`** - SVD and ALS. Use `scipy` or `surprise`.
   Establish numbers to beat before moving to anything more complex. Done, on
   the `svd`/`als` branches.

7. **`notebooks/02_item2vec.ipynb`** - embedding-based model. Compare against
   baselines, especially on long-tail items. Done, on the `item-2-vec` branch.

8. **`notebooks/03_sequential.ipynb`** - LSTM, then SASRec, then BERT4Rec.
   Read the BERT4Rec paper and Attention Is All You Need before implementing.
   SASRec and BERT4Rec done, on the `transformer` branch; LSTM not yet trained.

9. **`notebooks/04_ensemble.ipynb`** - combine best models. Requires all previous
   notebooks to have stable, evaluated results. Not started.

Separately, and not part of this sequence: the `app` branch built an MVP web
app serving all five already-trained models and deployed it — see "MVP web
app & deployment" above.

---

## Git workflow

- `main` is protected. Never commit directly to it.
- One branch per feature: `feature/svd-baseline`, `fix/overlap-check`,
  `notebook/eda`, `feature/evaluation-metrics`.
- Every merge goes through a pull request reviewed by at least one teammate.
- Clear notebook outputs before every commit.
- Before committing, show the user the code changes and the result of running
  the relevant code/tests.
- `data/` is in `.gitignore`. Never commit dataset files.

In practice, day-to-day work on the `app` branch (the MVP app + deployment)
commits directly to that branch rather than going through a feature
branch/PR per change — the user reviews and approves each diff in
conversation before every commit, and separately before every push. Apply
that lighter loop there; keep the feature-branch/PR flow above for the
research branches (`svd`, `als`, `item-2-vec`, `transformer`) and for
anything merging into `main`.

---

## Practical rules

- Pin package versions after installing anything new: `pip freeze > requirements.txt`
- Upload the Amazon dataset to the shared Drive folder once; everyone symlinks to it.
- `README.md` must always have working setup instructions - anyone should be able
  to clone and run in under 10 minutes.
- Primary metrics are NDCG@10 and Recall@10. Always report both.
- All results go in `results/` with enough context to know which model produced them.
