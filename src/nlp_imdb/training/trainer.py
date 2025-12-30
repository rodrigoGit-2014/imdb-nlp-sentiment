# src/nlp_imdb/training/trainer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nlp_imdb.data.dataset_loader import load_dataset_from_config
from nlp_imdb.training.metrics import compute_classification_metrics

# Modelo A
from nlp_imdb.models.model_a_tfidf import ModelAConfig, ModelATfidf
from nlp_imdb.features.tfidf import TfidfConfig

# Tokenización/encoding para RNN (Modelo B)
from nlp_imdb.preprocessing.tokenization import (
    RnnTokenizationConfig,
    build_vocab,
    encode_rnn,
)


@dataclass(frozen=True)
class TrainResult:
    validation: dict[str, float]
    test: dict[str, float]


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------
# Modelo A (TF-IDF + LogReg)
# -------------------------
def train_model_a_from_config(cfg: dict[str, Any]) -> TrainResult:
    # 1) dataset
    contract = load_dataset_from_config(cfg)
    dsd = contract.dataset

    text_col = contract.text_col
    label_col = contract.label_col

    train_ds = dsd["train"]
    val_ds = dsd["validation"]
    test_ds = dsd["test"]

    X_train = [str(x) for x in train_ds[text_col]]
    y_train = [int(x) for x in train_ds[label_col]]

    X_val = [str(x) for x in val_ds[text_col]]
    y_val = [int(x) for x in val_ds[label_col]]

    X_test = [str(x) for x in test_ds[text_col]]
    y_test = [int(x) for x in test_ds[label_col]]

    # 2) model config
    model_cfg = cfg.get("model", {})
    tfidf_cfg = model_cfg.get("tfidf", {})
    clf_cfg = model_cfg.get("classifier", {})

    tfidf = TfidfConfig(
        max_features=int(tfidf_cfg.get("max_features", 50_000)),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        min_df=int(tfidf_cfg.get("min_df", 2)),
        max_df=float(tfidf_cfg.get("max_df", 0.95)),
    )

    mcfg = ModelAConfig(
        tfidf=tfidf,
        C=float(clf_cfg.get("C", 2.0)),
        max_iter=int(clf_cfg.get("max_iter", 2000)),
        class_weight=clf_cfg.get("class_weight", "balanced"),
    )

    model = ModelATfidf(mcfg)

    # 3) train
    model.fit(X_train, y_train)

    # 4) eval
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = compute_classification_metrics(y_val, val_pred).as_dict()
    test_metrics = compute_classification_metrics(y_test, test_pred).as_dict()

    # 5) save artifacts
    artifacts_dir = cfg.get("artifacts", {}).get("dir", "artifacts/model_a")
    out_dir = _ensure_dir(artifacts_dir)

    model.save(out_dir)

    (out_dir / "metrics.validation.json").write_text(
        json.dumps(val_metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.test.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )

    return TrainResult(validation=val_metrics, test=test_metrics)


# ----------------------------------------
# Modelo B (RNN + Embeddings estáticos)
# ----------------------------------------
def train_model_b_from_config(cfg: dict[str, Any]) -> TrainResult:
    """
    Entrena un modelo RNN simple con embeddings estáticos (PyTorch).
    Nota: este trainer asume que tienes torch instalado.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Para Modelo B necesitas PyTorch instalado. "
            "Si estás en Python 3.13 es común que pip no encuentre torch. "
            "Recomendación: crea venv con Python 3.11/3.12 e instala torch ahí."
        ) from e

    # 1) dataset
    contract = load_dataset_from_config(cfg)
    dsd = contract.dataset
    text_col = contract.text_col
    label_col = contract.label_col

    train_ds = dsd["train"]
    val_ds = dsd["validation"]
    test_ds = dsd["test"]

    X_train = [str(x) for x in train_ds[text_col]]
    y_train = [int(x) for x in train_ds[label_col]]

    X_val = [str(x) for x in val_ds[text_col]]
    y_val = [int(x) for x in val_ds[label_col]]

    X_test = [str(x) for x in test_ds[text_col]]
    y_test = [int(x) for x in test_ds[label_col]]

    # 2) configs
    model_cfg = cfg.get("model", {})
    tok_cfg = model_cfg.get("tokenization", {})
    rnn_cfg = model_cfg.get("rnn", {})
    train_cfg = cfg.get("training", {})

    tok = RnnTokenizationConfig(
        max_vocab=int(tok_cfg.get("max_vocab", 40_000)),
        min_freq=int(tok_cfg.get("min_freq", 2)),
        max_length=int(tok_cfg.get("max_length", 256)),
        pad_token=str(tok_cfg.get("pad_token", "<PAD>")),
        unk_token=str(tok_cfg.get("unk_token", "<UNK>")),
    )

    embedding_dim = int(rnn_cfg.get("embedding_dim", 100))
    hidden_size = int(rnn_cfg.get("hidden_size", 128))
    num_layers = int(rnn_cfg.get("num_layers", 1))
    bidirectional = bool(rnn_cfg.get("bidirectional", True))
    dropout = float(rnn_cfg.get("dropout", 0.2))

    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("lr", 1e-3))
    epochs = int(train_cfg.get("epochs", 3))
    device = str(train_cfg.get("device", "cpu"))

    # 3) vocab + encoding
    vocab = build_vocab(X_train, cfg=tok)
    Xtr = encode_rnn(X_train, vocab=vocab, cfg=tok)
    Xva = encode_rnn(X_val, vocab=vocab, cfg=tok)
    Xte = encode_rnn(X_test, vocab=vocab, cfg=tok)

    # tensores
    Xtr_t = torch.tensor(Xtr, dtype=torch.long)
    ytr_t = torch.tensor(y_train, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.long)
    yva_t = torch.tensor(y_val, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.long)
    yte_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(Xva_t, yva_t), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(Xte_t, yte_t), batch_size=batch_size, shuffle=False
    )

    # 4) modelo (simple y mantenible)
    class RnnClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(
                num_embeddings=len(vocab), embedding_dim=embedding_dim, padding_idx=0
            )
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            out_dim = hidden_size * (2 if bidirectional else 1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(out_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x = self.emb(x)  # (B, T, E)
            # _, h = self.rnn(x)  # h: (num_layers * num_dirs, B, H)
            # h_last = h[-1]  # último bloque
            # h_last = self.dropout(h_last)
            # logits = self.fc(h_last).squeeze(-1)
            # return logits
            x = self.emb(x)  # (B, T, E)
            _, h = self.rnn(x)  # h: (num_layers * num_dirs, B, H)
            if bidirectional:
                # concatena forward y backward del último layer => (B, 2H)
                h_last = torch.cat([h[-2], h[-1]], dim=1)
            else:
                h_last = h[-1]  # (B, H)
            h_last = self.dropout(h_last)
            logits = self.fc(h_last).squeeze(-1)
            return logits

    model = RnnClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def _predict(loader: DataLoader) -> list[int]:
        model.eval()
        preds: list[int] = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                preds.extend((probs >= 0.5).long().cpu().tolist())
        return preds

    # 5) train loop
    for _epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    # 6) eval
    val_pred = _predict(val_loader)
    test_pred = _predict(test_loader)

    val_metrics = compute_classification_metrics(y_val, val_pred).as_dict()
    test_metrics = compute_classification_metrics(y_test, test_pred).as_dict()

    # 7) save artifacts
    artifacts_dir = cfg.get("artifacts", {}).get("dir", "artifacts/model_b")
    out_dir = _ensure_dir(artifacts_dir)

    torch.save(model.state_dict(), out_dir / "model_b.pt")
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.validation.json").write_text(
        json.dumps(val_metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.test.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )

    return TrainResult(validation=val_metrics, test=test_metrics)
