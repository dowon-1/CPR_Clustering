"""Cluster-level descriptive and statistical analysis."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import mannwhitneyu


def audit_dataframe(df: pd.DataFrame, redact_values: bool = True) -> pd.DataFrame:
    """Summarize missingness and unique-value counts.

    For public release, redact_values=True hides all numeric summaries.
    """

    result = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100,
        "unique_value_count": df.nunique(dropna=False),
    })
    if redact_values:
        return result.astype(object).where(result.isna(), "REDACTED")
    return result


def cluster_size_table(
    df: pd.DataFrame,
    cluster_column: str = "Cluster_Labels",
    redact_values: bool = True,
) -> pd.DataFrame:
    """Return cluster-size counts or a redacted schema."""

    counts = df[cluster_column].value_counts().sort_index().rename("n").to_frame()
    if redact_values:
        counts["n"] = "REDACTED"
    return counts


def cluster_outcome_summary(
    df: pd.DataFrame,
    target_column: str,
    cluster_column: str = "Cluster_Labels",
    redact_values: bool = True,
) -> pd.DataFrame:
    """Summarize outcome counts by cluster."""

    rows = []
    for label, group in df.groupby(cluster_column):
        total = int(len(group))
        positive = int((group[target_column] == 1).sum()) if target_column in group.columns else np.nan
        negative = int((group[target_column] == 0).sum()) if target_column in group.columns else np.nan
        rate = float(positive / total * 100) if total and not pd.isna(positive) else np.nan
        rows.append({
            cluster_column: label,
            "total": total,
            "target_positive": positive,
            "target_negative": negative,
            "target_positive_percent": rate,
        })
    out = pd.DataFrame(rows).set_index(cluster_column)
    if redact_values:
        return out.astype(object).where(out.isna(), "REDACTED")
    return out


def summarize_numeric_features_by_cluster(
    df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    cluster_column: str = "Cluster_Labels",
    redact_values: bool = True,
) -> pd.DataFrame:
    """Calculate mean and standard deviation for numeric variables by cluster."""

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != cluster_column and is_numeric_dtype(df[c])]

    grouped = df.groupby(cluster_column)[list(feature_columns)].agg(["mean", "std"])
    if redact_values:
        return grouped.astype(object).where(grouped.isna(), "REDACTED")
    return grouped


def mann_whitney_cluster_tests(
    df: pd.DataFrame,
    cluster_column: str = "Cluster_Labels",
    excluded_columns: Sequence[str] = (),
    alpha: float = 0.05,
    redact_values: bool = True,
) -> pd.DataFrame:
    """Compare each cluster with all other samples using Mann-Whitney U tests."""

    excluded = set(excluded_columns) | {cluster_column}
    unique_clusters = sorted(df[cluster_column].dropna().unique())
    rows = []

    for column in df.columns:
        if column in excluded or not is_numeric_dtype(df[column]):
            continue
        for label in unique_clusters:
            cluster_values = df.loc[df[cluster_column] == label, column].dropna()
            other_values = df.loc[df[cluster_column] != label, column].dropna()
            if len(cluster_values) == 0 or len(other_values) == 0:
                p_value = np.nan
                mean_value = np.nan
                std_value = np.nan
            elif pd.concat([cluster_values, other_values]).nunique() <= 1:
                p_value = np.nan
                mean_value = float(cluster_values.mean())
                std_value = float(cluster_values.std())
            else:
                _, p_value = mannwhitneyu(cluster_values, other_values, alternative="two-sided")
                mean_value = float(cluster_values.mean())
                std_value = float(cluster_values.std())

            rows.append({
                "variable": column,
                "cluster": label,
                "mean": mean_value,
                "std": std_value,
                "p_value": p_value,
                "significant": bool(pd.notna(p_value) and p_value < alpha),
            })

    out = pd.DataFrame(rows)
    if redact_values and not out.empty:
        for column in ["mean", "std", "p_value", "significant"]:
            out[column] = "REDACTED"
    return out
