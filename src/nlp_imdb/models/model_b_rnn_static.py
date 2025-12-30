# src/nlp_imdb/models/model_b_rnn_static.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelBConfig:
    # Embedding
    embedding_dim: int = 100
    freeze_embeddings: bool = True

    # RNN
    rnn_type: str = "lstm"  # "lstm" o "gru"
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.2  # dropout entre layers (si num_layers>1) y/o en classifier

    # Clasificador
    classifier_dropout: float = 0.2


class ModelBRnnStatic(nn.Module):
    """
    RNN (LSTM/GRU) + embeddings estáticos (matriz pre-armada).
    Output: logits (float) para BCEWithLogitsLoss.
    """

    def __init__(
        self, *, embedding_matrix: torch.Tensor, pad_idx: int, cfg: ModelBConfig
    ):
        super().__init__()
        self.cfg = cfg
        vocab_size, emb_dim = embedding_matrix.shape
        if emb_dim != cfg.embedding_dim:
            raise ValueError(
                f"embedding_dim mismatch: matrix={emb_dim} cfg={cfg.embedding_dim}"
            )

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx,
        )
        self.embedding.weight = nn.Parameter(
            embedding_matrix, requires_grad=not cfg.freeze_embeddings
        )

        rnn_cls = nn.LSTM if cfg.rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        rnn_out_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.dropout = nn.Dropout(cfg.classifier_dropout)
        self.fc = nn.Linear(rnn_out_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T] (long)
        returns logits: [B] (float)
        """
        x = self.embedding(input_ids)  # [B, T, E]
        out, h = self.rnn(x)  # out: [B, T, H*dir]

        # Tomamos último hidden state de la última capa
        if isinstance(h, tuple):  # LSTM => (h_n, c_n)
            h_n = h[0]
        else:  # GRU => h_n
            h_n = h

        # h_n: [num_layers*dir, B, H]
        if self.cfg.bidirectional:
            # última capa: forward y backward
            forward_last = h_n[-2]  # [B, H]
            backward_last = h_n[-1]  # [B, H]
            feat = torch.cat([forward_last, backward_last], dim=1)  # [B, 2H]
        else:
            feat = h_n[-1]  # [B, H]

        feat = self.dropout(feat)
        logits = self.fc(feat).squeeze(1)  # [B]
        return logits
