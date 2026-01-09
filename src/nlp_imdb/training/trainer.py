# src/nlp_imdb/training/trainer.py
from __future__ import annotations

from nlp_imdb.training.result import TrainResult
from nlp_imdb.training.train_a import train_model_a_from_config
from nlp_imdb.training.train_b import train_model_b_from_config
from nlp_imdb.training.train_c import train_model_c_from_config

__all__ = [
    "TrainResult",
    "train_model_a_from_config",
    "train_model_b_from_config",
    "train_model_c_from_config",
]
