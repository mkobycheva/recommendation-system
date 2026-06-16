"""Recommendation metrics shared by notebooks."""

from collections.abc import Mapping


def ndcg_at_k(*args, **kwargs):
    raise NotImplementedError("ndcg_at_k is not implemented yet.")


def recall_at_k(*args, **kwargs):
    raise NotImplementedError("recall_at_k is not implemented yet.")


def precision_at_k(*args, **kwargs):
    raise NotImplementedError("precision_at_k is not implemented yet.")


def rmse(*args, **kwargs):
    raise NotImplementedError("rmse is not implemented yet.")


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
