# src/nlp_imdb/data/dataset_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class DatasetContract:
    """
    Contract mínimo para esta tarea (alineado a dataset_contract.md):

    - splits: train / validation / test
    - columnas: text (str), label (int: 0=neg, 1=pos)
    """

    dataset: DatasetDict
    text_col: str = "text"
    label_col: str = "label"


def _ensure_columns(ds: Dataset, text_col: str, label_col: str) -> Dataset:
    cols = set(ds.column_names)
    if text_col not in cols:
        raise ValueError(f"Missing text column '{text_col}'. Available: {sorted(cols)}")
    if label_col not in cols:
        raise ValueError(f"Missing label column '{label_col}'. Available: {sorted(cols)}")
    return ds


def load_imdb_hf(
    *,
    cache_dir: str | Path,
    seed: int,
    validation_ratio: float,
    stratify: bool,
    text_col: str = "text",
    label_col: str = "label",
) -> DatasetContract:
    """
    Carga IMDb desde Hugging Face Datasets y crea split 'validation' desde 'train'.

    IMDb (HF) trae: train/test, con columnas típicas: 'text' y 'label' (0/1).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dsd: DatasetDict = load_dataset("imdb", cache_dir=str(cache_dir))

    if "train" not in dsd or "test" not in dsd:
        raise ValueError(f"Expected splits train/test in HF imdb. Got: {list(dsd.keys())}")

    train_full = _ensure_columns(dsd["train"], text_col, label_col)
    test = _ensure_columns(dsd["test"], text_col, label_col)

    split_kwargs: dict[str, Any] = {"test_size": validation_ratio, "seed": seed}
    if stratify:
        split_kwargs["stratify_by_column"] = label_col

    split = train_full.train_test_split(**split_kwargs)

    train = split["train"]
    validation = split["test"]

    out = DatasetDict(train=train, validation=validation, test=test)
    return DatasetContract(dataset=out, text_col=text_col, label_col=label_col)


def load_dataset_from_config(cfg: dict[str, Any]) -> DatasetContract:
    """
    Lee config con la estructura actual de configs/dataset.yaml:

    dataset:
      name: "imdb"
      source: "huggingface"
      text_field: "text"
      label_field: "label"

    splits:
      validation_ratio: 0.10
      stratify: true
      seed: 42
    """
    ds_cfg = cfg.get("dataset", {})
    splits_cfg = cfg.get("splits", {})

    name = ds_cfg.get("name", "imdb")
    source = ds_cfg.get("source", "huggingface")
    text_col = ds_cfg.get("text_field", "text")
    label_col = ds_cfg.get("label_field", "label")

    if name != "imdb":
        raise ValueError(f"Unsupported dataset name '{name}'. Only 'imdb' for now.")
    if source != "huggingface":
        raise ValueError(f"Unsupported dataset source '{source}'. Only 'huggingface' for now.")

    validation_ratio = float(splits_cfg.get("validation_ratio", 0.10))
    seed = int(splits_cfg.get("seed", 42))
    stratify = bool(splits_cfg.get("stratify", True))

    # cache_dir fijo por convención del proyecto
    cache_dir = "data/raw"

    return load_imdb_hf(
        cache_dir=cache_dir,
        seed=seed,
        validation_ratio=validation_ratio,
        stratify=stratify,
        text_col=text_col,
        label_col=label_col,
    )
