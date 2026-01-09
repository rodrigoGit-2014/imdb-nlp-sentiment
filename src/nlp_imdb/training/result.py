# src/nlp_imdb/training/result.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainResult:
    validation: dict[str, float]
    test: dict[str, float]
