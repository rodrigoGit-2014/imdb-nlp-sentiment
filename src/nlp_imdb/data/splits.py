# src/nlp_imdb/data/splits.py

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_validation_split(
    df: pd.DataFrame,
    validation_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide un DataFrame en train y validation de forma estratificada.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=validation_ratio,
        random_state=seed,
        stratify=df["label"],
    )

    return train_df, val_df
