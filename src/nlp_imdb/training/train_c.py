# src/nlp_imdb/training/train_c.py
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from nlp_imdb.data.dataset_loader import load_dataset_from_config
from nlp_imdb.training.metrics import compute_classification_metrics
from nlp_imdb.training.result import TrainResult


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def train_model_c_from_config(cfg: dict[str, Any]) -> TrainResult:
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as e:
        raise RuntimeError("Para Modelo C necesitas transformers instalado.") from e

    # 1) dataset (contract)
    contract = load_dataset_from_config(cfg)
    dsd = contract.dataset

    text_col = contract.text_col
    label_col = contract.label_col

    model_cfg = cfg.get("model", {})
    pretrained_name = str(model_cfg.get("pretrained_name", "distilbert-base-uncased"))
    max_length = int(model_cfg.get("max_length", 256))

    tr_cfg = cfg.get("training", {})
    batch_size = int(tr_cfg.get("batch_size", 16))
    eval_batch_size = int(tr_cfg.get("eval_batch_size", 32))
    epochs = int(tr_cfg.get("epochs", 2))
    lr = float(tr_cfg.get("lr", 2e-5))
    weight_decay = float(tr_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(tr_cfg.get("warmup_ratio", 0.06))
    logging_steps = int(tr_cfg.get("logging_steps", 50))

    # 2) tokenizer + tokenization
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=True)

    def _tok(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dsd.map(_tok, batched=True)

    # Ensure labels column name expected by Trainer
    if label_col != "labels":
        tokenized = tokenized.rename_column(label_col, "labels")

    # Trainer doesn't like stray columns that can't be tensorized
    keep_cols = {"input_ids", "attention_mask", "labels"}
    for split in tokenized.keys():
        cols = set(tokenized[split].column_names)
        drop = [c for c in cols if c not in keep_cols]
        if drop:
            tokenized[split] = tokenized[split].remove_columns(drop)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3) model
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, num_labels=2)

    # 4) metrics must be dict[str,float]
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        m = compute_classification_metrics(labels, preds)
        # ensure dict output
        return m.as_dict() if hasattr(m, "as_dict") else dict(m)

    # 5) training args (supports both eval_strategy & evaluation_strategy)
    out_dir = cfg.get("artifacts", {}).get("dir", "artifacts/model_c")
    artifacts_dir = _ensure_dir(out_dir)

    kwargs = dict(
        output_dir=str(artifacts_dir / "hf_runs"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_strategy=str(tr_cfg.get("save_strategy", "epoch")),
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"

    args = TrainingArguments(**kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 6) final eval
    val_out = trainer.predict(tokenized["validation"])
    test_out = trainer.predict(tokenized["test"])

    val_metrics_raw = val_out.metrics
    test_metrics_raw = test_out.metrics

    def _pick(m: dict[str, Any], prefix: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in ["accuracy", "precision", "recall", "f1"]:
            kk = f"{prefix}_{k}"
            if kk in m:
                out[k] = float(m[kk])
        return out

    val_metrics = _pick(val_metrics_raw, "test") or _pick(val_metrics_raw, "eval") or _pick(val_metrics_raw, "validation") or {}
    test_metrics = _pick(test_metrics_raw, "test") or _pick(test_metrics_raw, "eval") or {}

    # 7) save model + tokenizer + metrics
    model_dir = artifacts_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    (artifacts_dir / "metrics.validation.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    (artifacts_dir / "metrics.test.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    return TrainResult(validation=val_metrics, test=test_metrics)
