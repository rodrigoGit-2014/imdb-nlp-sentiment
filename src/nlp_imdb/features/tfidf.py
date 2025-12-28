# src/nlp_imdb/features/tfidf.py

from typing import Iterable, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


class TfidfFeatureExtractor:
    """
    TF-IDF feature extractor wrapper.
    """

    def __init__(
        self,
        max_features: int = 30000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
        )

    def fit(self, texts: Iterable[str]) -> None:
        self.vectorizer.fit(texts)

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: Iterable[str]) -> csr_matrix:
        return self.vectorizer.fit_transform(texts)
