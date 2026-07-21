# Cross-Domain Recommender System (Books + Movies)

A recommender system built on the Amazon Reviews 2023 dataset, exploring one
core question: can a model trained on a user's ratings in one domain (books)
reliably predict their preferences in another (movies)? Built as a course
project, with each model developed on its own branch and evaluated on shared
metrics before an MVP web app tied them together for hands-on use.

## Live demo

**[recsys-app-323823900175.europe-central2.run.app](https://recsys-app-323823900175.europe-central2.run.app)**

Search a book or movie, add a few to a cart, and get cold-start
recommendations in the other domain from any of five trained models. This is
a temporary course-project deployment (Google Cloud Run, scales to zero when
idle — the first request after a quiet period can take a minute or two to
wake up) and isn't meant to stay up indefinitely. See [DEPLOY.md](DEPLOY.md)
for how it's deployed.

## Models

Five models, each developed on its own branch, sharing the same evaluation
harness (MAP@5, NDCG@10, Recall@10, plus Top-Popular/Random baselines):

| Model    | Branch        | Approach                                    |
|----------|---------------|----------------------------------------------|
| SVD      | `svd`         | Matrix factorization, cold-start fold-in      |
| ALS      | `als`         | Matrix factorization, cold-start fold-in      |
| item2vec | `item-2-vec`  | Word2Vec over interaction sequences, averaged vectors |
| SASRec   | `transformer` | Causal (unidirectional) transformer           |
| BERT4Rec | `transformer` | Bidirectional transformer, Cloze-style masking |
| LSTM     | `lstm`        | In progress                                   |

Cold-start recommendations work by folding a cart of items the model never
trained on into its embedding space — no retraining needed per user.

## What's in this repo

- **`notebooks/`, `src/`, `configs/`** — the research pipeline: data prep,
  matrix building, evaluation metrics, and one notebook per modeling stage.
  This is the primary deliverable; see [AGENTS.md](AGENTS.md) for the full
  structure and development order.
- **`app/`** — MVP web app (FastAPI backend + Gradio frontend) serving
  cold-start cross-domain recommendations from all five trained models. See
  [app/README.md](app/README.md) to run it locally.
- **`space/main.py`** — single-process version of the same app, packaged for
  deployment (Cloud Run / HF Spaces expect one process, not the FastAPI +
  Gradio split `app/` uses locally).
- **`Dockerfile`, `DEPLOY.md`** — how the live demo above is built and
  deployed to Google Cloud Run.

## Data

Source: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io),
domains Books and Movies & TV. Users/items are 5-core filtered; splits are
time-based (last interaction per user = test, second-to-last = validation).

## Setup (research pipeline)

```bash
pip install -r requirements.txt
```

Run the overlap check first — verifies enough users exist across both
domains to make cross-domain transfer viable:

```bash
python3 src/data/overlap_check.py
```

Data lives under `data/raw/` (gitignored, not committed) — see
[AGENTS.md](AGENTS.md) for the shared Google Drive / Colab setup this
project uses.

## Running the MVP app locally

See [app/README.md](app/README.md) — includes a synthetic-artifacts script
so it runs without any real trained model.
