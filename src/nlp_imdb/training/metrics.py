# src/nlp_imdb/training/metrics.py

from typing import Dict, Iterable

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
