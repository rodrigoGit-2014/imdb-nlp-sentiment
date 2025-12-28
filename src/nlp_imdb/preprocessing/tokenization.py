# src/nlp_imdb/preprocessing/tokenization.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class WhitespaceTokenizeConfig:
    min_token_len: int = 1


def whitespace_tokenize(text: str, cfg: WhitespaceTokenizeConfig = WhitespaceTokenizeConfig()) -> List[str]:
    """
    Simple baseline tokenization (whitespace split).
    This is NOT used for BERT tokenization (transformers handles that).
    """
    tokens = text.split()
    if cfg.min_token_len <= 1:
        return tokens
    return [t for t in tokens if len(t) >= cfg.min_token_len]


def batch_whitespace_tokenize(
    texts: Sequence[str],
    cfg: WhitespaceTokenizeConfig = WhitespaceTokenizeConfig(),
) -> List[List[str]]:
    return [whitespace_tokenize(t, cfg) for t in texts]
