# src/nlp_imdb/features/embeddings_static.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


EmbeddingProvider = Literal["torchtext_glove"]


@dataclass(frozen=True)
class StaticEmbeddingConfig:
    provider: EmbeddingProvider = "torchtext_glove"
    name: str = "6B"  # torchtext: "6B" o "42B" (según disponibilidad)
    dim: int = 100
    freeze: bool = True
    cache_dir: str = "data/raw/embeddings"


def _load_torchtext_glove(name: str, dim: int, cache_dir: str):
    """
    Carga GloVe vía torchtext. Descarga si no existe en cache_dir.
    """
    try:
        from torchtext.vocab import GloVe  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Falta torchtext. Instala torchtext compatible con tu torch.\n"
            "Ej: pip install torchtext"
        ) from e

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    return GloVe(name=name, dim=dim, cache=str(cache))


def build_embedding_matrix(
    vocab: dict[str, int],
    cfg: StaticEmbeddingConfig,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
) -> torch.Tensor:
    """
    Construye matriz [vocab_size, dim] alineada a vocab token->id.

    - PAD queda en vector 0
    - UNK queda aleatorio pequeño
    - tokens presentes en GloVe usan su vector
    """
    vocab_size = len(vocab)

    if cfg.provider != "torchtext_glove":
        raise ValueError(f"Unsupported embedding provider: {cfg.provider}")

    glove = _load_torchtext_glove(cfg.name, cfg.dim, cfg.cache_dir)

    emb = torch.empty((vocab_size, cfg.dim), dtype=torch.float32)
    emb.uniform_(-0.05, 0.05)

    # PAD -> zeros
    if pad_token in vocab:
        emb[vocab[pad_token]].zero_()

    # UNK -> random (ya está random)
    _ = vocab.get(unk_token, None)

    found = 0
    for tok, idx in vocab.items():
        if tok in (pad_token, unk_token):
            continue
        if tok in glove.stoi:
            emb[idx] = glove.vectors[glove.stoi[tok]]
            found += 1

    coverage = found / max(vocab_size - 2, 1)
    print(
        f"[embeddings] coverage={coverage:.2%} ({found}/{vocab_size-2}) using GloVe {cfg.name}.{cfg.dim}d"
    )

    return emb
