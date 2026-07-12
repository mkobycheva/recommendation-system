"""Cold-start inference for the sequential baselines (SASRec, BERT4Rec).

Both treat the cart as a pseudo-sequence in the order items were selected --
there is no real interaction timestamp for a brand-new user, so this is a
known approximation of the training-time sequences, not a faithful one.
"""

import json
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from app.recommenders.base import Recommender


class _TransformerBackbone(nn.Module):
    """Item+position embedding -> TransformerEncoder -> LayerNorm, shared by
    SASRec (causal) and BERT4Rec (bidirectional)."""

    def __init__(self, vocab_size, max_len, d_model=64, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.max_len = max_len

    def _encode(self, item_seq, causal: bool):
        batch_size, seq_len = item_seq.shape
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_emb(item_seq) + self.pos_emb(positions)

        padding_mask = item_seq == 0
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(item_seq.device) if causal else None
        x = self.encoder(x, mask=mask, src_key_padding_mask=padding_mask)

        x = torch.nan_to_num(x, nan=0.0)
        return self.layer_norm(x)


class SASRecModel(_TransformerBackbone):
    def __init__(self, num_items, max_len, d_model=64, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__(num_items, max_len, d_model, n_heads, n_layers, dropout)

    def forward(self, item_seq):
        return self._encode(item_seq, causal=True)


class BERT4RecModel(_TransformerBackbone):
    def __init__(self, num_items, mask_token, max_len, d_model=64, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__(num_items + 1, max_len, d_model, n_heads, n_layers, dropout)
        self.num_items = num_items
        self.mask_token = mask_token

    def forward(self, item_seq):
        return self._encode(item_seq, causal=False)

    def real_item_embeddings(self):
        """Embedding matrix excluding the [MASK] row -- it must never be a candidate."""
        return self.item_emb.weight[: self.num_items]


class SequentialRecommender(Recommender):
    def __init__(self, model, item2idx, domain_item_indices, max_len):
        self.model = model
        self.model.eval()
        self.item2idx = item2idx
        self.idx2item = {idx: item_id for item_id, idx in item2idx.items()}
        self.domain_item_indices = domain_item_indices
        self.max_len = max_len

    @staticmethod
    def _load_common(artifacts_dir):
        artifacts_dir = Path(artifacts_dir)
        config = json.loads((artifacts_dir / "config.json").read_text())
        item2idx = json.loads((artifacts_dir / "item2idx.json").read_text())
        domain_item_indices = json.loads((artifacts_dir / "domain_item_indices.json").read_text())
        state_dict = torch.load(artifacts_dir / "model_state_dict.pt", map_location="cpu", weights_only=True)
        return config, item2idx, domain_item_indices, state_dict

    @abstractmethod
    def _build_input_sequence(self, item_indices: list[int]) -> list[int]:
        """Turn known selected-item indices into a left-padded model input of length max_len."""
        raise NotImplementedError

    @abstractmethod
    def _score_embeddings(self) -> torch.Tensor:
        """Embedding matrix to score candidates against (row i = item index i)."""
        raise NotImplementedError

    @torch.no_grad()
    def recommend(self, selected_items: list[str], target_domain: str, k: int = 10) -> list[str]:
        known_idx = [self.item2idx[item_id] for item_id in selected_items if item_id in self.item2idx]
        if not known_idx:
            raise ValueError(f"None of the selected items are known to this model: {selected_items!r}")

        candidates = self.domain_item_indices.get(target_domain, [])
        if not candidates:
            return []

        seq = self._build_input_sequence(known_idx)
        hidden = self.model(torch.tensor([seq], dtype=torch.long))[0, -1, :]

        embeddings = self._score_embeddings()
        candidate_tensor = torch.tensor(candidates, dtype=torch.long)
        scores = (embeddings[candidate_tensor] @ hidden).numpy()

        selected_idx = set(known_idx)
        mask = np.fromiter((idx in selected_idx for idx in candidates), dtype=bool, count=len(candidates))
        scores = np.where(mask, -np.inf, scores)

        top_n = min(k, int(np.isfinite(scores).sum()))
        if top_n == 0:
            return []

        top_positions = np.argpartition(-scores, top_n - 1)[:top_n]
        top_positions = top_positions[np.argsort(-scores[top_positions])]

        return [self.idx2item[candidates[int(pos)]] for pos in top_positions]


class SASRecRecommender(SequentialRecommender):
    @classmethod
    def load(cls, artifacts_dir) -> "SASRecRecommender":
        config, item2idx, domain_item_indices, state_dict = cls._load_common(artifacts_dir)
        model = SASRecModel(
            num_items=config["num_items"],
            max_len=config["max_len"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
        )
        model.load_state_dict(state_dict)
        return cls(model, item2idx, domain_item_indices, config["max_len"])

    def _build_input_sequence(self, item_indices):
        seq = item_indices[-self.max_len:]
        pad = self.max_len - len(seq)
        return [0] * pad + seq

    def _score_embeddings(self):
        return self.model.item_emb.weight


class BERT4RecRecommender(SequentialRecommender):
    def __init__(self, model, item2idx, domain_item_indices, max_len, mask_token):
        super().__init__(model, item2idx, domain_item_indices, max_len)
        self.mask_token = mask_token

    @classmethod
    def load(cls, artifacts_dir) -> "BERT4RecRecommender":
        config, item2idx, domain_item_indices, state_dict = cls._load_common(artifacts_dir)
        model = BERT4RecModel(
            num_items=config["num_items"],
            mask_token=config["mask_token"],
            max_len=config["max_len"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
        )
        model.load_state_dict(state_dict)
        return cls(model, item2idx, domain_item_indices, config["max_len"], config["mask_token"])

    def _build_input_sequence(self, item_indices):
        history = item_indices[-(self.max_len - 1):]
        seq = history + [self.mask_token]
        pad = self.max_len - len(seq)
        return [0] * pad + seq

    def _score_embeddings(self):
        return self.model.real_item_embeddings()
