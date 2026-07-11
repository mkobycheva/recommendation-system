"""Non-personalized baseline recommenders for comparison against trained models.

Both factories are compatible with the interface consumed by
``evaluate_multi_domain``: ``func(user_ids, target_domain, k, **kwargs) ->
dict[user_id, list[item_id]]``, and with the ``domain_item_indices`` /
``idx2item`` structures returned by ``src.data.build_matrix.build_user_item_matrix``.
"""

import numpy as np


def make_popularity_recommender(train_df, domain_item_indices, idx2item):
    """Recommend the same globally most-popular items of the target domain to everyone.

    Popularity is measured by interaction count in ``train_df``.
    """
    domain_items = {
        domain: {idx2item[idx] for idx in indices}
        for domain, indices in domain_item_indices.items()
    }
    popularity_counts = train_df["item_id"].value_counts()

    top_items_by_domain = {}
    for domain, items in domain_items.items():
        domain_counts = popularity_counts[popularity_counts.index.isin(items)]
        top_items_by_domain[domain] = domain_counts.index.tolist()

    def recommend(user_ids, target_domain, k=10, **kwargs):
        top_items = top_items_by_domain.get(target_domain, [])[:k]
        return {user_id: list(top_items) for user_id in user_ids}

    return recommend


def make_random_recommender(domain_item_indices, idx2item, seed=42):
    """Recommend k random items of the target domain, deterministic per seed."""
    domain_items = {
        domain: np.array([idx2item[idx] for idx in indices])
        for domain, indices in domain_item_indices.items()
    }
    rng = np.random.default_rng(seed)

    def recommend(user_ids, target_domain, k=10, **kwargs):
        pool = domain_items.get(target_domain, np.array([]))
        recommendations = {}
        for user_id in sorted(user_ids, key=str):
            sample_size = min(k, len(pool))
            if sample_size == 0:
                recommendations[user_id] = []
            else:
                recommendations[user_id] = rng.choice(pool, size=sample_size, replace=False).tolist()
        return recommendations

    return recommend
