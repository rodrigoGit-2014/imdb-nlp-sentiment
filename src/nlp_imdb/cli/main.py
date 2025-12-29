# src/nlp_imdb/cli/main.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from nlp_imdb.data.dataset_loader import load_dataset_from_config


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _print_samples(dsd, split: str, n_pos: int = 3, n_neg: int = 3) -> None:
    ds = dsd[split]

    # Seleccionamos ejemplos por label (0=neg, 1=pos)
    pos = ds.filter(lambda x: int(x["label"]) == 1)
    neg = ds.filter(lambda x: int(x["label"]) == 0)

    print("\n" + "=" * 80)
    print(f"Split: {split} | total={len(ds)} | pos={len(pos)} | neg={len(neg)}")
    print("=" * 80)

    print("\n--- 3 RESEÑAS POSITIVAS (label=1) ---")
    for i in range(min(n_pos, len(pos))):
        text = pos[i]["text"]
        print(f"\n[{i+1}] {text}")

    print("\n--- 3 RESEÑAS NEGATIVAS (label=0) ---")
    for i in range(min(n_neg, len(neg))):
        text = neg[i]["text"]
        print(f"\n[{i+1}] {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP IMDb Sentiment Analysis CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()

    print(f"Using config file: {args.config}")

    cfg = _read_yaml(args.config)

    contract = load_dataset_from_config(cfg)
    dsd = contract.dataset  # DatasetDict con train/validation/test

    print("\nLoaded dataset (contract-compliant):")
    for split_name in ["train", "validation", "test"]:
        if split_name in dsd:
            print(f"  - {split_name}: {len(dsd[split_name])} rows")

    # Muestra ejemplos para ver “resultado” al ejecutar dataset.yaml
    _print_samples(dsd, split="train", n_pos=3, n_neg=3)

    sys.exit(0)


if __name__ == "__main__":
    main()
