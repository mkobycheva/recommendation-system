# AGENTS.md - Recommender System Project

## What this project is

A cross-domain recommender system for books, movies, and music using the Amazon
Reviews 2023 dataset. The core research question: can a model trained on a user's
ratings in one domain reliably predict their preferences in another?

The approach is to establish baselines with simple models, then progressively build
toward sequential and ensemble methods, evaluating each on cross-domain transfer.

---

## Project structure

```text
recommender-system/
├── data/
│   ├── raw/              <- never modify; gitignored
│   ├── processed/
│   └── splits/
├── notebooks/
│   ├── 00_eda.ipynb           <- exploratory data analysis
│   ├── 01_baselines.ipynb     <- SVD and ALS
│   ├── 02_item2vec.ipynb
│   ├── 03_sequential.ipynb    <- LSTM, SASRec, BERT4Rec
│   └── 04_ensemble.ipynb
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
├── requirements.txt
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
that enough users exist across multiple domains to make cross-domain transfer viable.

**`src/evaluation/`** contains shared metric functions used by every notebook.
Write this before training any model. Primary metrics are NDCG@10 and Recall@10.
Secondary metrics are Precision@10 and RMSE.
MAP@K must be implemented here as shared Python code, not only inside a notebook.

**`configs/model_configs.py`** stores all hyperparameters as plain Python dicts.
Never hardcode hyperparameters inside a notebook. Import them from here instead.

**`data/`** is gitignored. Never commit the Amazon dataset. Raw files go in
`data/raw/` and are never modified. Processed outputs go in `data/processed/`.
Train/val/test splits go in `data/splits/`.

**`results/`** stores metric outputs and plots produced by notebooks.

---

## Current ALS baseline scope

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

---

## Data

Source: Amazon Reviews 2023 - https://amazon-reviews-2023.github.io

Domains used: Books, Movies & TV, CDs & Vinyl.

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
!wget -q https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/CDs_and_Vinyl.jsonl.gz
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

Work through these in sequence. Each step unblocks the next.

1. **`src/data/overlap_check.py`** - load all three domains, find users appearing
   in 2+ domains, report count. If overlap is below ~10k users, revisit sampling.

2. **`src/data/preprocess.py` + `build_matrix.py`** - clean ratings, build sparse
   user-item matrices per domain, apply 5-core filtering.

3. **`src/data/train_test_split.py`** - time-based leave-one-out split.

4. **`src/evaluation/metrics.py`** - implement NDCG@10, Recall@10, Precision@10,
   RMSE, and `cross_domain_eval.py` for train-on-A / test-on-B evaluation.

5. **`notebooks/00_eda.ipynb`** - explore the data: rating distributions, user
   activity, item popularity, overlap counts, sparsity per domain.

6. **`notebooks/01_baselines.ipynb`** - SVD and ALS. Use `scipy` or `surprise`.
   Establish numbers to beat before moving to anything more complex.

7. **`notebooks/02_item2vec.ipynb`** - embedding-based model. Compare against
   baselines, especially on long-tail items.

8. **`notebooks/03_sequential.ipynb`** - LSTM, then SASRec, then BERT4Rec.
   Read the BERT4Rec paper and Attention Is All You Need before implementing.

9. **`notebooks/04_ensemble.ipynb`** - combine best models. Requires all previous
   notebooks to have stable, evaluated results.

---

## Git workflow

- `main` is protected. Never commit directly to it.
- One branch per feature: `feature/svd-baseline`, `fix/overlap-check`,
  `notebook/eda`, `feature/evaluation-metrics`.
- Every merge goes through a pull request reviewed by at least one teammate.
- Clear notebook outputs before every commit.
- `data/` is in `.gitignore`. Never commit dataset files.

---

## Practical rules

- Pin package versions after installing anything new: `pip freeze > requirements.txt`
- Upload the Amazon dataset to the shared Drive folder once; everyone symlinks to it.
- `README.md` must always have working setup instructions - anyone should be able
  to clone and run in under 10 minutes.
- Primary metrics are NDCG@10 and Recall@10. Always report both.
- All results go in `results/` with enough context to know which model produced them.
