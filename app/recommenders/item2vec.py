"""Cold-start inference for the item2vec (gensim Word2Vec) baseline."""

import json
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from app.recommenders.base import Recommender


class Item2VecRecommender(Recommender):
    """Cold-start recommender via averaged item2vec vectors of selected items.

    domain_item_indices holds positions into model.wv.vectors / model.wv.index_to_key
    (gensim's own vocabulary order) -- not the src.data.build_matrix indexing, which
    is a separate, unrelated vocabulary.
    """

    def __init__(self, model: Word2Vec, domain_item_indices: dict):
        self.model = model
        self.domain_item_indices = domain_item_indices

    @classmethod
    def load(cls, artifacts_dir) -> "Item2VecRecommender":
        artifacts_dir = Path(artifacts_dir)
        model = Word2Vec.load(str(artifacts_dir / "word2vec.model"))
        domain_item_indices = json.loads((artifacts_dir / "domain_item_indices.json").read_text())
        return cls(model, domain_item_indices)

    def recommend(self, selected_items: list[str], target_domain: str, k: int = 10) -> list[str]:
        known = [item_id for item_id in selected_items if item_id in self.model.wv]
        if not known:
            raise ValueError(f"None of the selected items are known to this model: {selected_items!r}")

        user_vector = np.mean([self.model.wv[item_id] for item_id in known], axis=0)

        candidate_positions = np.asarray(self.domain_item_indices.get(target_domain, []), dtype=np.int64)
        if candidate_positions.size == 0:
            return []

        candidate_vectors = self.model.wv.vectors[candidate_positions]
        scores = candidate_vectors @ user_vector

        selected_positions = {self.model.wv.key_to_index[item_id] for item_id in known}
        mask = np.fromiter(
            (pos in selected_positions for pos in candidate_positions), dtype=bool, count=len(candidate_positions)
        )
        scores = np.where(mask, -np.inf, scores)

        top_n = min(k, int(np.isfinite(scores).sum()))
        if top_n == 0:
            return []

        top_order = np.argpartition(-scores, top_n - 1)[:top_n]
        top_order = top_order[np.argsort(-scores[top_order])]

        return [self.model.wv.index_to_key[int(candidate_positions[pos])] for pos in top_order]
