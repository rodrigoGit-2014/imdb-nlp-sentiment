# src/nlp_imdb/models/base.py

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """
    Base interface for all models in the project.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any) -> None:
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
