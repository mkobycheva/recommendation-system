"""Recommendation metrics shared by notebooks."""

from collections.abc import Mapping
import numpy as np


def ndcg_at_k(recommended, relevant, k=10):
    """Normalized Discounted Cumulative Gain at K."""
    relevant = set(relevant)
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(recommended, relevant, k=10):
    """Recall at K."""
    relevant = set(relevant)
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def precision_at_k(recommended, relevant, k=10):
    """Precision at K."""
    relevant = set(relevant)
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k if k > 0 else 0.0


def rmse(predicted, actual):
    """Root Mean Squared Error."""
    predicted = np.array(predicted)
    actual = np.array(actual)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def average_precision_at_k(recommended_items, relevant_items, k=10):
    """Compute average precision at K for one user.

    Duplicate recommendations are removed after their first occurrence before
    applying the ranking cutoff.
    Users with no relevant held-out items receive 0.0.
    """
    if k <= 0:
        return 0.0

    relevant = set(relevant_items or [])
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen = set()
    deduped_recommendations = []

    for item in recommended_items or []:
        if item in seen:
            continue
        seen.add(item)
        deduped_recommendations.append(item)
        if len(deduped_recommendations) == k:
            break

    for rank, item in enumerate(deduped_recommendations, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(len(relevant), k)


def map_at_k(recommended_items_by_user, relevant_items_by_user, k=10):
    """Compute mean average precision at K across users.

    Args:
        recommended_items_by_user: Mapping of user_id to ordered recommended item IDs.
        relevant_items_by_user: Mapping of user_id to relevant held-out item IDs.
        k: Ranking cutoff.

    Returns:
        Mean AP@K over users in ``relevant_items_by_user``. Users with no relevant
        items contribute 0.0.
    """
    if not isinstance(recommended_items_by_user, Mapping):
        raise TypeError("recommended_items_by_user must be a mapping")
    if not isinstance(relevant_items_by_user, Mapping):
        raise TypeError("relevant_items_by_user must be a mapping")

    if not relevant_items_by_user:
        return 0.0

    average_precisions = [
        average_precision_at_k(
            recommended_items_by_user.get(user_id, []),
            relevant_items,
            k=k,
        )
        for user_id, relevant_items in relevant_items_by_user.items()
    ]
    return sum(average_precisions) / len(average_precisions)
