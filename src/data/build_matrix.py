"""Sparse user-item matrix construction utilities."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class InteractionMatrix:
    """Shared indexes and sparse train matrix for recommender baselines."""

    user_item_train: csr_matrix
    user2idx: dict
    item2idx: dict
    idx2item: dict
    item_domain: dict
    train_seen_idx_by_user: dict
    domain_item_indices: dict


def add_domain_item_ids(df, domain, item_col="parent_asin"):
    """Return a copy with domain and globally unique item_id columns."""
    df = df.copy()
    df["domain"] = domain
    df["item_id"] = domain + "::" + df[item_col].astype(str)
    return df


def build_user_item_matrix(
    train_df,
    domains=None,
    user_col="user_id",
    item_col="item_id",
    value_col=None,
):
    """Build shared indexes and a sparse user-item matrix.

    Args:
        train_df: Training interactions with user, item_id, and domain columns.
        domains: Optional iterable of domains to index for domain-filtered
            recommendations. Defaults to all domains present in train_df.
        user_col: User identifier column.
        item_col: Globally unique item identifier column.
        value_col: Optional value column. When omitted, every observed
            interaction receives value 1.0 for implicit-feedback models.

    Returns:
        InteractionMatrix containing the sparse train matrix and lookup maps.
    """
    if "domain" not in train_df.columns:
        raise ValueError("train_df must contain a 'domain' column")

    all_users = train_df[user_col].drop_duplicates().to_numpy()
    all_items = train_df[item_col].drop_duplicates().to_numpy()

    user2idx = {user_id: idx for idx, user_id in enumerate(all_users)}
    item2idx = {item_id: idx for idx, item_id in enumerate(all_items)}
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}

    item_domains = train_df[[item_col, "domain"]].drop_duplicates(subset=item_col)
    item_domain = {
        item2idx[row[item_col]]: row["domain"]
        for _, row in item_domains.iterrows()
    }

    user_indices = train_df[user_col].map(user2idx).to_numpy()
    item_indices = train_df[item_col].map(item2idx).to_numpy()
    if value_col is not None:
        values = train_df[value_col].to_numpy(dtype=np.float32)
    else:
        values = np.ones(len(train_df), dtype=np.float32)

    user_item_train = csr_matrix(
        (values, (user_indices, item_indices)),
        shape=(len(user2idx), len(item2idx)),
    )
    user_item_train.sum_duplicates()

    train_seen_idx_by_user = {
        user_idx: set(user_item_train[user_idx].indices)
        for user_idx in range(user_item_train.shape[0])
    }

    if domains is None:
        domains = train_df["domain"].drop_duplicates().to_list()

    domain_item_indices = {
        domain: np.array(
            [idx for idx, item_dom in item_domain.items() if item_dom == domain],
            dtype=np.int64,
        )
        for domain in domains
    }

    return InteractionMatrix(
        user_item_train=user_item_train,
        user2idx=user2idx,
        item2idx=item2idx,
        idx2item=idx2item,
        item_domain=item_domain,
        train_seen_idx_by_user=train_seen_idx_by_user,
        domain_item_indices=domain_item_indices,
    )


def build_implicit_als_matrix(train_df, domains=None, user_col="user_id", item_col="item_id"):
    """Build a binary implicit-feedback matrix for ALS."""
    return build_user_item_matrix(
        train_df,
        domains=domains,
        user_col=user_col,
        item_col=item_col,
        value_col=None,
    )


def build_explicit_svd_matrix(
    train_df,
    domains=None,
    user_col="user_id",
    item_col="item_id",
    rating_col="rating",
):
    """Build a rating-valued sparse matrix for explicit-feedback SVD."""
    return build_user_item_matrix(
        train_df,
        domains=domains,
        user_col=user_col,
        item_col=item_col,
        value_col=rating_col,
    )
