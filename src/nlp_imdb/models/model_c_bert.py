# src/nlp_imdb/models/model_c_bert.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import BaseModel


class BertSentimentModel(BaseModel):
    """
    Model C: BERT fine-tuning for sentiment classification.
    """

    def __init__(self, pretrained_model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, num_labels=2
        )

    def fit(self, X, y) -> None:
        raise NotImplementedError("Training handled by Hugging Face Trainer.")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**X)
            return outputs.logits.argmax(dim=-1)

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
