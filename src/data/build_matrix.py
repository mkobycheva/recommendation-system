"""Sparse user-item matrix construction utilities for ALS and SVD pipelines."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
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
    user_col="user_id",
    item_col="item_id",
    mode="implicit",
):
    """Build shared indexes and a sparse user-item matrix from interaction data."""
    if "domain" not in train_df.columns:
        raise ValueError("train_df must contain a 'domain' column")

    # Deduplicate before matrix generation to protect matrix weights from corrupting
    if mode == "explicit":
        # SVD: Keep the latest rating to preserve chronologically accurate feedback
        unique_df = train_df.sort_values("timestamp").drop_duplicates(
            subset=[user_col, item_col], keep="last"
        )
    else:
        # ALS: Keep the highest rating to capture the strongest preference confidence
        unique_df = train_df.sort_values("rating").drop_duplicates(
            subset=[user_col, item_col], keep="last"
        )

    # Generate sequential integer vocabularies
    all_users = unique_df[user_col].drop_duplicates().to_numpy()
    all_items = unique_df[item_col].drop_duplicates().to_numpy()

    user2idx = {user_id: idx for idx, user_id in enumerate(all_users)}
    item2idx = {item_id: idx for idx, item_id in enumerate(all_items)}
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}

    # Optimized string splitting to instantly extract domains from item IDs
    item_domain = {idx: item_id.split("::")[0] for item_id, idx in item2idx.items()}

    # Map text strings directly into parallel integer coordinate index arrays
    user_indices = unique_df[user_col].map(user2idx).to_numpy(dtype=np.int32)
    item_indices = unique_df[item_col].map(item2idx).to_numpy(dtype=np.int32)

    # Directly convert the explicit rating column into your payload matrix values
    values = unique_df["rating"].to_numpy(dtype=np.float32)

    # Build the Compressed Sparse Row matrix container
    user_item_train = csr_matrix(
        (values, (user_indices, item_indices)),
        shape=(len(user2idx), len(item2idx)),
    )
    user_item_train.sum_duplicates()

    # Track historical interaction items per user to mask out during recommendation
    train_seen_idx_by_user = {
        user_idx: set(user_item_train[user_idx].indices)
        for user_idx in range(user_item_train.shape[0])
    }

    # Dynamically discover domain groups from data instead of hardcoding lists
    unique_domains = unique_df["domain"].drop_duplicates().to_list()
    domain_item_indices = {
        domain: np.array(
            [idx for idx, item_dom in item_domain.items() if item_dom == domain],
            dtype=np.int64,
        )
        for domain in unique_domains
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


def build_implicit_als_matrix(train_df, user_col="user_id", item_col="item_id"):
    """Build a continuous preference-confidence matrix using raw explicit star values."""
    return build_user_item_matrix(
        train_df,
        user_col=user_col,
        item_col=item_col,
        mode="implicit",
    )


def build_explicit_svd_matrix(
    train_df,
    user_col="user_id",
    item_col="item_id",
):
    """Build a rating-valued sparse matrix for explicit-feedback centered SVD."""
    return build_user_item_matrix(
        train_df,
        user_col=user_col,
        item_col=item_col,
        mode="explicit",
    )