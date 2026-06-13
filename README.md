# Recommender System

Cross-domain recommender system for books, movies, and music using the Amazon
Reviews 2023 dataset.

## Setup

```bash
pip install -r requirements.txt
```

Run the overlap check first:

```bash
python3 src/data/overlap_check.py
```

Data should live under `data/raw/`, but the `data/` directory is gitignored.

