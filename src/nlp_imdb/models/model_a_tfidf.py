# src/nlp_imdb/models/model_a_tfidf.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from nlp_imdb.features.tfidf import TfidfConfig, build_vectorizer


@dataclass(frozen=True)
class ModelAConfig:
    tfidf: TfidfConfig
    C: float = 2.0
    max_iter: int = 2000
    class_weight: str | None = "balanced"


class ModelATfidf:
    """
    TF-IDF + Logistic Regression (modelo clásico).
    """

    def __init__(self, cfg: ModelAConfig) -> None:
        self.cfg = cfg
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("tfidf", build_vectorizer(cfg.tfidf)),
                (
                    "clf",
                    LogisticRegression(
                        C=cfg.C,
                        max_iter=cfg.max_iter,
                        class_weight=cfg.class_weight,
                        n_jobs=None,
                    ),
                ),
            ]
        )

    def fit(self, X_text: list[str], y: list[int]) -> None:
        self.pipeline.fit(X_text, y)

    def predict(self, X_text: list[str]) -> list[int]:
        return self.pipeline.predict(X_text).tolist()

    def predict_proba(self, X_text: list[str]) -> list[Tuple[float, float]]:
        proba = self.pipeline.predict_proba(X_text)
        return [tuple(map(float, row)) for row in proba]

    def save(self, artifacts_dir: str | Path) -> None:
        p = Path(artifacts_dir)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, p / "model.joblib")

    @staticmethod
    def load(artifacts_dir: str | Path) -> "ModelATfidf":
        p = Path(artifacts_dir)
        pipeline = joblib.load(p / "model.joblib")
        obj = object.__new__(ModelATfidf)  # type: ignore
        obj.cfg = None  # type: ignore
        obj.pipeline = pipeline  # type: ignore
        return obj
