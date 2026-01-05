# src/nlp_imdb/cli/main.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from datasets import DatasetDict

from nlp_imdb.data.dataset_loader import load_dataset_from_config
from nlp_imdb.preprocessing.text_cleaning import get_default_config
from nlp_imdb.preprocessing.tokenization import clean_texts
from nlp_imdb.training.trainer import (
    train_model_a_from_config,
    train_model_b_from_config,
    train_model_c_from_config,
)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _print_samples(
    dsd: DatasetDict, split: str, n_pos: int = 3, n_neg: int = 3
) -> None:
    ds = dsd[split]
    pos = ds.filter(lambda x: int(x["label"]) == 1)
    neg = ds.filter(lambda x: int(x["label"]) == 0)

    print("\n" + "=" * 80)
    print(f"Split: {split} | total={len(ds)} | pos={len(pos)} | neg={len(neg)}")
    print("=" * 80)

    print(f"\n--- {n_pos} RESEÑAS POSITIVAS (label=1) ---")
    for i in range(min(n_pos, len(pos))):
        print(f"\n[{i+1}] {pos[i]['text']}")

    print(f"\n--- {n_neg} RESEÑAS NEGATIVAS (label=0) ---")
    for i in range(min(n_neg, len(neg))):
        print(f"\n[{i+1}] {neg[i]['text']}")


def _run_dataset(cfg: dict[str, Any]) -> None:
    contract = load_dataset_from_config(cfg)
    dsd = contract.dataset

    print("\nLoaded dataset (contract-compliant):")
    for split_name in ["train", "validation", "test"]:
        if split_name in dsd:
            print(f"  - {split_name}: {len(dsd[split_name])} rows")

    _print_samples(dsd, split="train", n_pos=3, n_neg=3)


def _run_preprocess(cfg: dict[str, Any]) -> None:
    contract = load_dataset_from_config(cfg)
    dsd = contract.dataset

    prep_cfg = cfg.get("preprocessing", {})
    profile = str(prep_cfg.get("profile", "classic"))
    cleaning_cfg = get_default_config(profile)  # usa TU text_cleaning.py estable

    def _clean_batch(batch: dict[str, Any]) -> dict[str, Any]:
        texts = batch["text"]
        cleaned = clean_texts(texts, profile=cleaning_cfg.profile, cfg=cleaning_cfg)
        return {"text": cleaned}

    dsd_clean = DatasetDict()
    for split_name in dsd.keys():
        dsd_clean[split_name] = dsd[split_name].map(
            _clean_batch, batched=True, batch_size=1000
        )

    out_dir = Path(prep_cfg.get("output_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # OJO: save_to_disk crea una CARPETA con *.arrow + metadata (no un solo archivo)
    out_path = out_dir / str(prep_cfg.get("output_name", "imdb_cleaned"))
    dsd_clean.save_to_disk(str(out_path))

    print(f"\n✅ Preprocess listo. Guardado en: {out_path}")
    print(
        "Contenido típico: data-00000-of-00001.arrow + dataset_info.json + state.json"
    )


def _run_train_a(cfg: dict[str, Any]) -> None:
    res = train_model_a_from_config(cfg)
    print("\n✅ Modelo A entrenado y evaluado.")
    print(f"Validation metrics: {res.validation}")
    print(f"Test metrics:       {res.test}")


def _run_train_b(cfg: dict[str, Any]) -> None:
    res = train_model_b_from_config(cfg)
    print("\n✅ Modelo B entrenado y evaluado.")
    print(f"Validation metrics: {res.validation}")
    print(f"Test metrics:       {res.test}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP IMDb Sentiment Analysis CLI")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["dataset", "preprocess", "train_a", "train_b", "train_c"],
        help="Pipeline stage to run",
    )
    args = parser.parse_args()

    cfg = _read_yaml(args.config)

    print(f"Using config file: {args.config}")
    print(f"Stage: {args.stage}")

    if args.stage == "dataset":
        _run_dataset(cfg)
    elif args.stage == "preprocess":
        _run_preprocess(cfg)
    elif args.stage == "train_a":
        _run_train_a(cfg)
    elif args.stage == "train_b":
        _run_train_b(cfg)
    elif args.stage == "train_c":
        res = train_model_c_from_config(cfg)
        print("\n✅ Modelo C entrenado y evaluado.")
        print(f"Validation metrics: {res.validation}")
        print(f"Test metrics:       {res.test}")

    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    sys.exit(0)


if __name__ == "__main__":
    main()
