# src/features/tfidf.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TfidfConfig:
    max_features: int = 50_000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95


def build_vectorizer(cfg: TfidfConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
    )
