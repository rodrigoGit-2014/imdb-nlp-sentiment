# src/nlp_imdb/training/train_b.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import json
from pathlib import Path
from typing import Any

from nlp_imdb.data.dataset_loader import load_dataset_from_config
from nlp_imdb.preprocessing.text_cleaning import clean_text
#from nlp_imdb.preprocessing.tokenization import build_vocab, encode_texts_for_rnn
from nlp_imdb.preprocessing.tokenization import RnnTokenizationConfig, build_vocab, encode_rnn

from nlp_imdb.training.metrics import compute_classification_metrics
from nlp_imdb.training.result import TrainResult


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def train_model_b_from_config(cfg: dict[str, Any]) -> TrainResult:


    # 1) dataset
    contract = load_dataset_from_config(cfg)
    dsd = contract.dataset

    text_col = contract.text_col
    label_col = contract.label_col

    train_ds = dsd["train"]
    val_ds = dsd["validation"]
    test_ds = dsd["test"]

    X_train_raw = [str(x) for x in train_ds[text_col]]
    y_train = [int(x) for x in train_ds[label_col]]

    X_val_raw = [str(x) for x in val_ds[text_col]]
    y_val = [int(x) for x in val_ds[label_col]]

    X_test_raw = [str(x) for x in test_ds[text_col]]
    y_test = [int(x) for x in test_ds[label_col]]

    # 2) preprocessing (aligned to your pipeline)
    X_train = [clean_text(t) for t in X_train_raw]
    X_val = [clean_text(t) for t in X_val_raw]
    X_test = [clean_text(t) for t in X_test_raw]

    # 3) config
    model_cfg = cfg.get("model", {})
    rnn_cfg = model_cfg.get("rnn", {})
    training_cfg = cfg.get("training", {})

    max_vocab = int(cfg.get("tokenization", {}).get("max_vocab", 30_000))
    max_length = int(cfg.get("tokenization", {}).get("max_length", 256))

    hidden_size = int(rnn_cfg.get("hidden_size", 128))
    embedding_dim = int(rnn_cfg.get("embedding_dim", 100))
    num_layers = int(rnn_cfg.get("num_layers", 1))
    bidirectional = bool(rnn_cfg.get("bidirectional", True))
    dropout = float(rnn_cfg.get("dropout", 0.2))

    batch_size = int(training_cfg.get("batch_size", 64))
    epochs = int(training_cfg.get("epochs", 3))
    lr = float(training_cfg.get("lr", 1e-3))
    device_str = str(training_cfg.get("device", "cpu")).lower().strip()
    device = torch.device(device_str)

    # 4) vocab + encoding (usar contract de tokenization.py)
    tok = RnnTokenizationConfig(
        max_vocab=max_vocab,
        min_freq=int(cfg.get("tokenization", {}).get("min_freq", 2)),
        max_length=max_length,
    )

    vocab = build_vocab(X_train, cfg=tok)
    X_train_ids = encode_rnn(X_train, vocab=vocab, cfg=tok)
    X_val_ids = encode_rnn(X_val, vocab=vocab, cfg=tok)
    X_test_ids = encode_rnn(X_test, vocab=vocab, cfg=tok)

    class _SeqDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return (
                torch.tensor(self.X[idx], dtype=torch.long),
                torch.tensor(self.y[idx], dtype=torch.long),
            )

    train_loader = DataLoader(_SeqDataset(X_train_ids, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_SeqDataset(X_val_ids, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_SeqDataset(X_test_ids, y_test), batch_size=batch_size, shuffle=False)

    # 5) model
    class RnnClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            out_dim = hidden_size * (2 if bidirectional else 1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(out_dim, 1)

        def forward(self, x):
            x = self.emb(x)
            _, h = self.rnn(x)

            # IMPORTANT: handle bidirectional correctly
            if bidirectional:
                h_last = torch.cat([h[-2], h[-1]], dim=1)
            else:
                h_last = h[-1]

            h_last = self.dropout(h_last)
            logits = self.fc(h_last).squeeze(-1)
            return logits

    model = RnnClassifier().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 6) train loop
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())

        print(f"Epoch {epoch}/{epochs} | train_loss={total / max(len(train_loader), 1):.4f}")

    # 7) predict
    def _predict(loader):
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                logits = model(xb)
                p = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
                preds.extend(p)
        return preds

    val_pred = _predict(val_loader)
    test_pred = _predict(test_loader)

    val_metrics = compute_classification_metrics(y_val, val_pred).as_dict()
    test_metrics = compute_classification_metrics(y_test, test_pred).as_dict()

    # 8) save artifacts
    artifacts_dir = cfg.get("artifacts", {}).get("dir", "artifacts/model_b")
    out_dir = _ensure_dir(artifacts_dir)

    torch.save(model.state_dict(), out_dir / "model_b.pt")
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.validation.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    (out_dir / "metrics.test.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    return TrainResult(validation=val_metrics, test=test_metrics)
