"""Train/validation splitting helpers."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_validation_split(
    df: pd.DataFrame,
    target_column: str,
    validation_size: float = 0.24,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a train/validation split with optional target stratification."""

    stratify_values = df[target_column] if stratify and target_column in df.columns else None
    train_index, valid_index = train_test_split(
        df.index,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_values,
    )
    return df.loc[train_index].copy(), df.loc[valid_index].copy()
