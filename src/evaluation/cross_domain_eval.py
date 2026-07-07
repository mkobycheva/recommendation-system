"""Unified cross-domain evaluation utilities for recommender baselines."""

from collections.abc import Mapping
import numpy as np
import pandas as pd
from src.evaluation.metrics import ndcg_at_k, recall_at_k, map_at_k


def relevant_items_by_user(split_df, target_domain):
    """Extract ground-truth target items per user for a specific domain."""
    domain_rows = split_df[split_df["domain"] == target_domain]
    return domain_rows.groupby("user_id")["item_id"].agg(set).to_dict()


def evaluate_multi_domain(
    split_dfs: dict[str, pd.DataFrame],
    recommend_func,
    k=10,
    **kwargs
):
    """
    Evaluates a recommendation function across multiple data splits and domains.

    Args:
        split_dfs: Dictionary of datasets, e.g., {'valid': valid_df, 'test': test_df}
        recommend_func: Inference function matching signature:
                        func(user_ids, target_domain, k, **kwargs) -> dict[user_id, list[item_id]]
        k: Top-N ranking cutoff parameter.
        **kwargs: Optional backend model factors or vocabularies passed through to inference loops.

    Returns:
        tuple: (validation_results_dict, test_results_dict)
    """
    if not isinstance(split_dfs, Mapping):
        raise TypeError("split_dfs must be a mapping of split names to DataFrames")

    all_results = {}
    target_domains = ["books", "movies"]

    for target_domain in target_domains:
        # Collect ground truth across all requested splits
        relevants = {
            name: relevant_items_by_user(df, target_domain)
            for name, df in split_dfs.items()
        }

        # Flatten distinct user IDs to process inference scoring in a single unified matrix pass
        all_users = list({u for r in relevants.values() for u in r})
        if not all_users:
            continue

        print(f"[Evaluation] Scoring {target_domain.upper()}: {len(all_users):,} unique users across splits")

        # Generate target recommendations using the pipeline's native vector model
        all_recs = recommend_func(all_users, target_domain, k=k, **kwargs)

        # Separate outputs back out into their respective split metrics arrays
        for split_name, relevant in relevants.items():
            if not relevant:
                continue

            recs = {u: all_recs.get(u, []) for u in relevant}

            # Compute parallel lists of performance metrics
            ndcg_scores = [ndcg_at_k(recs[u], relevant[u], k) for u in relevant if relevant[u]]
            recall_scores = [recall_at_k(recs[u], relevant[u], k) for u in relevant if relevant[u]]
            map_score = map_at_k(recs, relevant, k=k)

            all_results.setdefault(split_name, {})[target_domain] = {
                "ndcg@10": round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else 0.0,
                "recall@10": round(float(np.mean(recall_scores)), 4) if recall_scores else 0.0,
                "map@10": round(float(map_score), 4),
                "n_users": len(ndcg_scores),
            }

            print(f"  -> {split_name.upper()} Results:")
            print(f"     MAP@{k}: {all_results[split_name][target_domain]['map@10']:.4f} | "
                  f"NDCG@{k}: {all_results[split_name][target_domain]['ndcg@10']:.4f} | "
                  f"Recall@{k}: {all_results[split_name][target_domain]['recall@10']:.4f}")

    # Fallback to empty structures if a specific split was omitted during invocation
    valid_out = all_results.get("valid", {d: {} for d in target_domains})
    test_out = all_results.get("test", {d: {} for d in target_domains})

    return valid_out, test_out
