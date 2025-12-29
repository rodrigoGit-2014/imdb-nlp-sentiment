# src/nlp_imdb/data/splits.py
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_validation_split(
    df: pd.DataFrame,
    validation_ratio: float,
    seed: int,
    stratify: bool = True,
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide un DataFrame en train y validation.

    - stratify=True => estratifica por label_col
    """
    stratify_col = df[label_col] if stratify else None

    train_df, val_df = train_test_split(
        df,
        test_size=validation_ratio,
        random_state=seed,
        stratify=stratify_col,
    )
    return train_df, val_df
