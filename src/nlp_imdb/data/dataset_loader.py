# src/nlp_imdb/data/dataset_loader.py

from typing import Dict
from datasets import load_dataset


def load_imdb_dataset() -> Dict[str, object]:
    """
    Carga el dataset IMDb desde Hugging Face y devuelve
    los splits relevantes en formato raw.
    """
    dataset = load_dataset("imdb")

    return {
        "train": dataset["train"],
        "test": dataset["test"],
    }
