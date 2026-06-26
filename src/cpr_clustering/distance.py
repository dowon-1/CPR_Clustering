"""Distance-matrix utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gower_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute a simple mixed-type Gower distance matrix.

    Numeric columns are range-normalized. Categorical columns contribute 0 when
    values match and 1 when values differ. Missing values are ignored feature-wise.
    """

    if df.empty:
        return np.zeros((0, 0), dtype=float)

    n_samples = len(df)
    distance = np.zeros((n_samples, n_samples), dtype=float)
    valid_feature_count = np.zeros((n_samples, n_samples), dtype=float)

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            valid = ~np.isnan(values)
            if valid.sum() == 0:
                continue
            value_range = np.nanmax(values) - np.nanmin(values)
            if value_range == 0 or np.isnan(value_range):
                diff = np.zeros((n_samples, n_samples), dtype=float)
            else:
                diff = np.abs(values[:, None] - values[None, :]) / value_range
            pair_valid = valid[:, None] & valid[None, :]
        else:
            values = series.astype("object").where(series.notna(), None).to_numpy()
            valid = np.array([v is not None for v in values])
            diff = (values[:, None] != values[None, :]).astype(float)
            pair_valid = valid[:, None] & valid[None, :]

        distance += np.where(pair_valid, diff, 0.0)
        valid_feature_count += pair_valid.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.divide(distance, valid_feature_count, out=np.zeros_like(distance), where=valid_feature_count > 0)
    np.fill_diagonal(distance, 0.0)
    return distance
