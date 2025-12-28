# src/nlp_imdb/models/model_a_tfidf.py

from typing import Iterable
import joblib
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix

from .base import BaseModel


class TfidfLogisticRegressionModel(BaseModel):
    """
    Model A: TF-IDF + Logistic Regression.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 2000) -> None:
        self.model = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, X: csr_matrix, y: Iterable[int]) -> None:
        self.model.fit(X, y)

    def predict(self, X: csr_matrix):
        return self.model.predict(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
