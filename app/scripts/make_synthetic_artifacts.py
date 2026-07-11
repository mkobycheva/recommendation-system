"""Generate a small synthetic ARTIFACTS_DIR for local manual testing of the
backend/frontend without real Colab artifacts. Not used by the test suite
(tests build their own fixtures in-process) -- this is for `uvicorn`/`gradio`
manual runs, per app/README.md.

Usage: python -m app.scripts.make_synthetic_artifacts [output_dir]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

BOOK_ITEMS = [f"books::b{i}" for i in range(10)]
MOVIE_ITEMS = [f"movies::m{i}" for i in range(10)]
ALL_ITEMS = BOOK_ITEMS + MOVIE_ITEMS


def write_fold_in_artifacts(model_dir: Path, seed: int) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    item2idx = {item_id: idx for idx, item_id in enumerate(ALL_ITEMS)}
    rng = np.random.default_rng(seed)
    item_factors = rng.normal(size=(len(ALL_ITEMS), 16)).astype(np.float32)
    domain_item_indices = {
        "books": [item2idx[item_id] for item_id in BOOK_ITEMS],
        "movies": [item2idx[item_id] for item_id in MOVIE_ITEMS],
    }
    np.save(model_dir / "item_factors.npy", item_factors)
    (model_dir / "item2idx.json").write_text(json.dumps(item2idx))
    (model_dir / "domain_item_indices.json").write_text(json.dumps(domain_item_indices))


def write_item2vec_artifacts(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    sequences = [ALL_ITEMS] + [rng.choice(ALL_ITEMS, size=5, replace=False).tolist() for _ in range(50)]
    model = Word2Vec(sentences=sequences, vector_size=16, window=3, sg=1, min_count=1, workers=1, seed=3)
    model.save(str(model_dir / "word2vec.model"))
    domain_item_indices = {
        "books": [model.wv.key_to_index[i] for i in model.wv.index_to_key if i.startswith("books::")],
        "movies": [model.wv.key_to_index[i] for i in model.wv.index_to_key if i.startswith("movies::")],
    }
    (model_dir / "domain_item_indices.json").write_text(json.dumps(domain_item_indices))


def write_items_metadata(artifacts_dir: Path) -> None:
    rows = []
    for item_id in ALL_ITEMS:
        domain, asin = item_id.split("::")
        title_word = "Book" if domain == "books" else "Movie"
        rows.append({
            "item_id": item_id,
            "domain": domain,
            "title": f"Synthetic {title_word} {asin}",
            "image_url": None,
        })
    pd.DataFrame(rows).to_parquet(artifacts_dir / "items_metadata.parquet", index=False)


def main(output_dir: Path) -> None:
    write_fold_in_artifacts(output_dir / "svd", seed=1)
    write_fold_in_artifacts(output_dir / "als", seed=2)
    write_item2vec_artifacts(output_dir / "item2vec")
    write_items_metadata(output_dir)
    print(f"Synthetic artifacts written to {output_dir}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./artifacts")
    main(out)
