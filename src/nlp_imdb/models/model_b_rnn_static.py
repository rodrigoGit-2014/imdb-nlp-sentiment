# src/nlp_imdb/models/model_b_rnn_static.py

import torch
import torch.nn as nn

from .base import BaseModel


class StaticEmbeddingRNNModel(nn.Module, BaseModel):
    """
    Model B: Static Embeddings + RNN/LSTM.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def fit(self, X, y) -> None:
        raise NotImplementedError("Training handled by trainer.")

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            output, _ = self.rnn(X)
            logits = self.classifier(output[:, -1, :])
            return self.sigmoid(logits)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
