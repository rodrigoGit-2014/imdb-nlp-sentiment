# src/nlp_imdb/training/train_a.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nlp_imdb.data.dataset_loader import load_dataset_from_config
from nlp_imdb.features.tfidf import TfidfConfig
from nlp_imdb.models.model_a_tfidf import ModelAConfig, ModelATfidf
from nlp_imdb.training.metrics import compute_classification_metrics
from nlp_imdb.training.result import TrainResult


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_contract(cfg: dict[str, Any]):
    """Load dataset contract (dataset + column names) from config."""
    return load_dataset_from_config(cfg)


def _extract_splits(contract):
    """Extract train/validation/test splits from the DatasetDict."""
    dsd = contract.dataset
    return dsd["train"], dsd["validation"], dsd["test"]


def _extract_xy(split_ds, text_col: str, label_col: str) -> tuple[list[str], list[int]]:
    """Convert a HF split into python lists of texts and labels."""
    X = [str(x) for x in split_ds[text_col]]
    y = [int(x) for x in split_ds[label_col]]
    return X, y


def _build_model_config(cfg: dict[str, Any]) -> ModelAConfig:
    """Build ModelAConfig (TF-IDF + classifier params) from YAML config dict."""
    model_cfg = cfg.get("model", {})
    tfidf_cfg = model_cfg.get("tfidf", {})
    clf_cfg = model_cfg.get("classifier", {})

    tfidf = TfidfConfig(
        max_features=int(tfidf_cfg.get("max_features", 50_000)),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        min_df=int(tfidf_cfg.get("min_df", 2)),
        max_df=float(tfidf_cfg.get("max_df", 0.95)),
    )

    return ModelAConfig(
        tfidf=tfidf,
        C=float(clf_cfg.get("C", 2.0)),
        max_iter=int(clf_cfg.get("max_iter", 2000)),
        class_weight=clf_cfg.get("class_weight", "balanced"),
    )


def _train_model(
    mcfg: ModelAConfig, X_train: list[str], y_train: list[int]
) -> ModelATfidf:
    """Train Model A and return the trained model object."""
    model = ModelATfidf(mcfg)
    model.fit(X_train, y_train)
    return model


def _evaluate_model(model: ModelATfidf, X: list[str], y: list[int]) -> dict[str, float]:
    """Predict and compute metrics for a given split."""
    preds = model.predict(X)
    return compute_classification_metrics(y, preds).as_dict()


def _get_artifacts_dir(cfg: dict[str, Any]) -> str:
    """
    Read artifacts directory from config.
    Assumption: artifacts.dir MUST exist (no default fallback).
    """
    return cfg["artifacts"]["dir"]


def _save_artifacts(
    out_dir: Path,
    model: ModelATfidf,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    """Persist model and metrics to disk."""
    model.save(out_dir)
    (out_dir / "metrics.validation.json").write_text(
        json.dumps(val_metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.test.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )


def train_model_a_from_config(cfg: dict[str, Any]) -> TrainResult:
    # 1) load contract + splits
    contract = _load_contract(cfg)
    train_ds, val_ds, test_ds = _extract_splits(contract)

    # 2) extract X/y
    text_col = contract.text_col
    label_col = contract.label_col

    X_train, y_train = _extract_xy(train_ds, text_col=text_col, label_col=label_col)
    X_val, y_val = _extract_xy(val_ds, text_col=text_col, label_col=label_col)
    X_test, y_test = _extract_xy(test_ds, text_col=text_col, label_col=label_col)

    # 3) build config + train
    mcfg = _build_model_config(cfg)
    model = _train_model(mcfg, X_train=X_train, y_train=y_train)

    # 4) evaluate
    val_metrics = _evaluate_model(model, X=X_val, y=y_val)
    test_metrics = _evaluate_model(model, X=X_test, y=y_test)

    # 5) save artifacts
    artifacts_dir = _get_artifacts_dir(cfg)
    out_dir = _ensure_dir(artifacts_dir)
    _save_artifacts(
        out_dir, model=model, val_metrics=val_metrics, test_metrics=test_metrics
    )

    return TrainResult(validation=val_metrics, test=test_metrics)
