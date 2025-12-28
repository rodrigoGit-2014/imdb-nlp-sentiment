# src/nlp_imdb/training/trainer.py

from typing import Any, Iterable, Tuple

import numpy as np


class Trainer:
    """
    Generic trainer wrapper.

    Concrete training logic (sklearn / torch / HF Trainer)
    should be orchestrated from here or delegated to model-specific trainers.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def fit(self, X: Any, y: Iterable[int]) -> None:
        """
        Train the model.
        """
        self.model.fit(X, y)

    def predict(self, X: Any):
        """
        Run inference using the trained model.
        """
        return self.model.predict(X)

    def fit_predict(self, X_train: Any, y_train: Iterable[int], X_test: Any):
        """
        Convenience method: train and predict.
        """
        self.fit(X_train, y_train)
        return self.predict(X_test)
