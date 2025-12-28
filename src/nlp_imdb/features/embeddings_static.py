# src/nlp_imdb/features/embeddings_static.py

from typing import Dict, List
import numpy as np


class StaticEmbeddingLookup:
    """
    Simple static embedding lookup table.

    Expects a dictionary mapping tokens to embedding vectors.
    """

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.embedding_index: Dict[str, np.ndarray] = {}

    def add_embedding(self, token: str, vector: np.ndarray) -> None:
        if vector.shape[0] != self.embedding_dim:
            raise ValueError("Embedding dimension mismatch.")
        self.embedding_index[token] = vector

    def get_embedding(self, token: str) -> np.ndarray:
        return self.embedding_index.get(
            token, np.zeros(self.embedding_dim, dtype=np.float32)
        )

    def encode_sequence(self, tokens: List[str]) -> np.ndarray:
        """
        Converts a list of tokens into a 2D array [seq_len, embedding_dim].
        """
        return np.stack([self.get_embedding(t) for t in tokens], axis=0)
