"""Lift comparison between a model and non-personalized baselines."""

import pandas as pd

_METRIC_PREFIXES = [
    ("map@", "MAP@"),
    ("ndcg@", "NDCG@"),
    ("recall@", "Recall@"),
]


def _display_name(key):
    for prefix, label in _METRIC_PREFIXES:
        if key.startswith(prefix):
            return label + key[len(prefix):]
    return key


def compute_lift_table(model_results, popularity_results, random_results):
    """Build a Metric/Model/Top-Popular/Random/Lift table for one split+domain.

    Each argument is the dict returned per split+domain by evaluate_multi_domain
    (keys like "ndcg@10"/"recall@10"/"map@5", plus "n_users" which is ignored).
    """
    metric_keys = [
        key for prefix, _ in _METRIC_PREFIXES
        for key in model_results
        if key.startswith(prefix)
    ]

    rows = []
    for key in metric_keys:
        model_val = model_results[key]
        popular_val = popularity_results[key]
        random_val = random_results[key]

        lift_popular = model_val / popular_val if popular_val > 0 else float("inf")
        lift_random = model_val / random_val if random_val > 0 else float("inf")

        rows.append({
            "Metric": _display_name(key),
            "Model": model_val,
            "Top-Popular": popular_val,
            "Random": random_val,
            "Lift vs Popular": lift_popular,
            "Lift vs Random": lift_random,
        })

    return pd.DataFrame(rows)
