"""FastAPI backend: title search and cross-domain recommendations."""

from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rapidfuzz import fuzz, process

from app.config import get_items_metadata_path, get_model_dir
from app.recommenders.item2vec import Item2VecRecommender
from app.recommenders.sequential import BERT4RecRecommender, SASRecRecommender
from app.recommenders.svd_als import FoldInRecommender

MODEL_LOADERS = {
    "svd": FoldInRecommender.load,
    "als": FoldInRecommender.load,
    "item2vec": Item2VecRecommender.load,
    "sasrec": SASRecRecommender.load,
    "bert4rec": BERT4RecRecommender.load,
}


class ItemOut(BaseModel):
    item_id: str
    title: str
    domain: str
    image_url: str | None = None


class RecommendRequest(BaseModel):
    selected_items: list[str]
    target_domain: str
    model: Literal["svd", "als", "item2vec", "sasrec", "bert4rec"]
    k: int = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    items_metadata_path = get_items_metadata_path()
    if not items_metadata_path.exists():
        raise FileNotFoundError(
            f"items_metadata.parquet not found at {items_metadata_path}. "
            "Set ITEMS_METADATA_PATH or ARTIFACTS_DIR before starting the app."
        )
    app.state.items_metadata = pd.read_parquet(items_metadata_path)

    recommenders = {}
    for model_name, load in MODEL_LOADERS.items():
        model_dir = get_model_dir(model_name)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Artifacts for model {model_name!r} not found at {model_dir}. "
                "Set ARTIFACTS_DIR to a directory containing svd/als/item2vec/sasrec/bert4rec subfolders."
            )
        recommenders[model_name] = load(model_dir)
    app.state.recommenders = recommenders

    yield


app = FastAPI(lifespan=lifespan)


def _items_to_out(items_metadata: pd.DataFrame, item_ids: list[str]) -> list[dict]:
    results = items_metadata.set_index("item_id").reindex(item_ids).dropna(subset=["title"])
    results = results.reset_index()
    return results[["item_id", "title", "domain", "image_url"]].to_dict("records")


@app.get("/search", response_model=list[ItemOut])
def search(q: str, domain: Literal["books", "movies"]) -> list[dict]:
    items_metadata = app.state.items_metadata
    domain_items = items_metadata[items_metadata["domain"] == domain]
    if domain_items.empty or not q:
        return []

    choices = dict(zip(domain_items["item_id"], domain_items["title"]))
    matches = process.extract(q, choices, scorer=fuzz.WRatio, limit=10)
    matched_ids = [item_id for _, _score, item_id in matches]

    return _items_to_out(items_metadata, matched_ids)


@app.post("/recommend", response_model=list[ItemOut])
def recommend(request: RecommendRequest) -> list[dict]:
    recommender = app.state.recommenders[request.model]

    try:
        item_ids = recommender.recommend(request.selected_items, request.target_domain, k=request.k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _items_to_out(app.state.items_metadata, item_ids)
