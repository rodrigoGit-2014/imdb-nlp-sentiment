# src/nlp_imdb/training/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
        }


def compute_classification_metrics(
    y_true: list[int], y_pred: list[int]
) -> ClassificationMetrics:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return ClassificationMetrics(
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )
