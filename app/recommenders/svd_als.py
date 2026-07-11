"""Shared fold-in inference for SVD and ALS: same artifact shape, same formula."""

import json
from pathlib import Path

import numpy as np

from app.recommenders.base import Recommender


class FoldInRecommender(Recommender):
    """Cold-start recommender for factor models (SVD, ALS) via item-vector fold-in."""

    def __init__(self, item_factors: np.ndarray, item2idx: dict, domain_item_indices: dict):
        self.item_factors = item_factors
        self.item2idx = item2idx
        self.idx2item = {idx: item_id for item_id, idx in item2idx.items()}
        self.domain_item_indices = domain_item_indices

    @classmethod
    def load(cls, artifacts_dir) -> "FoldInRecommender":
        artifacts_dir = Path(artifacts_dir)
        item_factors = np.load(artifacts_dir / "item_factors.npy")
        item2idx = json.loads((artifacts_dir / "item2idx.json").read_text())
        domain_item_indices = json.loads((artifacts_dir / "domain_item_indices.json").read_text())
        return cls(item_factors, item2idx, domain_item_indices)

    def recommend(self, selected_items: list[str], target_domain: str, k: int = 10) -> list[str]:
        valid_idx = [self.item2idx[item_id] for item_id in selected_items if item_id in self.item2idx]
        if not valid_idx:
            raise ValueError(f"None of the selected items are known to this model: {selected_items!r}")

        user_vector = self.item_factors[valid_idx].sum(axis=0)
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm

        candidates = np.asarray(self.domain_item_indices.get(target_domain, []), dtype=np.int64)
        if candidates.size == 0:
            return []

        scores = self.item_factors[candidates] @ user_vector

        selected_idx = set(valid_idx)
        mask = np.fromiter((idx in selected_idx for idx in candidates), dtype=bool, count=len(candidates))
        scores = np.where(mask, -np.inf, scores)

        top_n = min(k, int(np.isfinite(scores).sum()))
        if top_n == 0:
            return []

        top_positions = np.argpartition(-scores, top_n - 1)[:top_n]
        top_positions = top_positions[np.argsort(-scores[top_positions])]

        return [self.idx2item[int(candidates[pos])] for pos in top_positions]
