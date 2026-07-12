import json

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from gensim.models import Word2Vec

from app.recommenders.sequential import BERT4RecModel, SASRecModel

BOOK_ITEMS = [f"books::b{i}" for i in range(6)]
MOVIE_ITEMS = [f"movies::m{i}" for i in range(6)]
ALL_ITEMS = BOOK_ITEMS + MOVIE_ITEMS
NUM_ITEMS = len(ALL_ITEMS) + 1  # +1 for padding
MAX_LEN = 8


def _write_fold_in_artifacts(model_dir, seed):
    model_dir.mkdir(parents=True)
    item2idx = {item_id: idx for idx, item_id in enumerate(ALL_ITEMS)}
    rng = np.random.default_rng(seed)
    item_factors = rng.normal(size=(len(ALL_ITEMS), 8)).astype(np.float32)
    domain_item_indices = {
        "books": [item2idx[item_id] for item_id in BOOK_ITEMS],
        "movies": [item2idx[item_id] for item_id in MOVIE_ITEMS],
    }
    np.save(model_dir / "item_factors.npy", item_factors)
    (model_dir / "item2idx.json").write_text(json.dumps(item2idx))
    (model_dir / "domain_item_indices.json").write_text(json.dumps(domain_item_indices))


def _write_item2vec_artifacts(model_dir):
    model_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    sequences = [ALL_ITEMS] + [rng.choice(ALL_ITEMS, size=4, replace=False).tolist() for _ in range(30)]
    model = Word2Vec(sentences=sequences, vector_size=8, window=3, sg=1, min_count=1, workers=1, seed=3)
    model.save(str(model_dir / "word2vec.model"))
    domain_item_indices = {
        "books": [model.wv.key_to_index[i] for i in model.wv.index_to_key if i.startswith("books::")],
        "movies": [model.wv.key_to_index[i] for i in model.wv.index_to_key if i.startswith("movies::")],
    }
    (model_dir / "domain_item_indices.json").write_text(json.dumps(domain_item_indices))


def _write_sequential_common(model_dir, model, config):
    torch.save(model.state_dict(), model_dir / "model_state_dict.pt")
    item2idx = {item_id: idx + 1 for idx, item_id in enumerate(ALL_ITEMS)}  # 0 = padding
    (model_dir / "item2idx.json").write_text(json.dumps(item2idx))
    domain_item_indices = {
        "books": [item2idx[item_id] for item_id in BOOK_ITEMS],
        "movies": [item2idx[item_id] for item_id in MOVIE_ITEMS],
    }
    (model_dir / "domain_item_indices.json").write_text(json.dumps(domain_item_indices))
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_sasrec_artifacts(model_dir):
    model_dir.mkdir(parents=True)
    torch.manual_seed(4)
    model = SASRecModel(num_items=NUM_ITEMS, max_len=MAX_LEN, d_model=8, n_heads=2, n_layers=1, dropout=0.0)
    _write_sequential_common(model_dir, model, {
        "num_items": NUM_ITEMS, "max_len": MAX_LEN,
        "d_model": 8, "n_heads": 2, "n_layers": 1, "dropout": 0.0,
    })


def _write_bert4rec_artifacts(model_dir):
    model_dir.mkdir(parents=True)
    torch.manual_seed(5)
    mask_token = NUM_ITEMS
    model = BERT4RecModel(
        num_items=NUM_ITEMS, mask_token=mask_token, max_len=MAX_LEN,
        d_model=8, n_heads=2, n_layers=1, dropout=0.0,
    )
    _write_sequential_common(model_dir, model, {
        "num_items": NUM_ITEMS, "max_len": MAX_LEN, "mask_token": mask_token,
        "d_model": 8, "n_heads": 2, "n_layers": 1, "dropout": 0.0,
    })


def _write_items_metadata(artifacts_dir):
    rows = []
    for item_id in ALL_ITEMS:
        domain, asin = item_id.split("::")
        rows.append({
            "item_id": item_id,
            "domain": domain,
            "title": f"Synthetic title for {asin}",
            "image_url": None,
        })
    pd.DataFrame(rows).to_parquet(artifacts_dir / "items_metadata.parquet", index=False)


@pytest.fixture
def artifacts_dir(tmp_path):
    _write_fold_in_artifacts(tmp_path / "svd", seed=1)
    _write_fold_in_artifacts(tmp_path / "als", seed=2)
    _write_item2vec_artifacts(tmp_path / "item2vec")
    _write_sasrec_artifacts(tmp_path / "sasrec")
    _write_bert4rec_artifacts(tmp_path / "bert4rec")
    _write_items_metadata(tmp_path)
    return tmp_path


@pytest.fixture
def client(artifacts_dir, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))

    from app.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_search_finds_known_title(client):
    response = client.get("/search", params={"q": "b0", "domain": "books"})

    assert response.status_code == 200
    results = response.json()
    assert any(item["item_id"] == "books::b0" for item in results)


def test_search_unknown_domain_is_rejected(client):
    response = client.get("/search", params={"q": "b0", "domain": "music"})

    assert response.status_code == 422


@pytest.mark.parametrize("model_name", ["svd", "als", "item2vec", "sasrec", "bert4rec"])
def test_recommend_returns_correct_shape(client, model_name):
    response = client.post(
        "/recommend",
        json={
            "selected_items": ["books::b0", "books::b1"],
            "target_domain": "movies",
            "model": model_name,
            "k": 5,
        },
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results) <= 5
    for item in results:
        assert set(item.keys()) == {"item_id", "title", "domain", "image_url"}
        assert item["domain"] == "movies"


def test_recommend_unknown_items_returns_400(client):
    response = client.post(
        "/recommend",
        json={
            "selected_items": ["books::totally_unknown"],
            "target_domain": "movies",
            "model": "svd",
            "k": 5,
        },
    )

    assert response.status_code == 400


def test_recommend_unknown_model_is_rejected(client):
    response = client.post(
        "/recommend",
        json={
            "selected_items": ["books::b0"],
            "target_domain": "movies",
            "model": "lstm",
            "k": 5,
        },
    )

    assert response.status_code == 422


def test_missing_artifacts_fail_at_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "does_not_exist"))

    from app.api.main import app

    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass
