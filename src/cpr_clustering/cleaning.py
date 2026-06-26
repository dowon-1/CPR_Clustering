"""Feature cleaning and preprocessing utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_VITAL_REPLACEMENTS: Dict[str, Mapping[float, float]] = {
    "initial": {-9: np.nan},
    "arrival": {999: np.nan},
}

VITAL_BIN_SPECS = {
    "pulse": {
        "columns": ["trmsppls", "ptmipuls"],
        "bins": [-np.inf, -1, 0, 29, 59, 100, 119, np.inf],
        "labels": ["N", "0", "1_29", "30_59", "60_100", "101_119", "_120"],
    },
    "systolic_bp": {
        "columns": ["trmspsbp", "ptmihibp", "trmslsbp"],
        "bins": [-np.inf, -1, 0, 49, 75, 89, np.inf],
        "labels": ["N", "0", "1_49", "50_75", "76_89", "_90"],
    },
    "diastolic_bp": {
        "columns": ["trmspdbp", "ptmilobp", "trmsldbp"],
        "bins": [-np.inf, -1, 0, 29, 45, 59, np.inf],
        "labels": ["N", "0", "1_29", "30_45", "46_59", "_60"],
    },
    "temperature": {
        "columns": ["trmspbdh", "ptmibdht"],
        "bins": [-np.inf, -1, 0, 24, 28, 32, 35, 37.8, np.inf],
        "labels": ["N", "0", "1_24", "24_1_28", "28_1_32", "32_1_35", "35_1_37_8", "_37_8"],
    },
    "spo2": {
        "columns": ["trmspoxs", "trmsvoxs"],
        "bins": [-np.inf, -1, 0, 80, 90, 95, np.inf],
        "labels": ["N", "0", "1_80", "81_90", "91_95", "_96"],
    },
    "respiratory_rate": {
        "columns": ["trmspbrt"],
        "bins": [-np.inf, -1, 0, 5, 9, 29, np.inf],
        "labels": ["N", "0", "1_5", "6_9", "10_29", "_30"],
    },
    "mean_arterial_pressure": {
        "columns": ["trmstmap"],
        "bins": [-np.inf, -1, 0, 36, 55, 69, np.inf],
        "labels": ["N", "0", "1_36", "37_55", "56_69", "_70"],
    },
}


def drop_existing_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Drop columns if they are present in a DataFrame."""

    return df.drop(columns=[c for c in columns if c in df.columns], errors="ignore")


def replace_dash_with_no(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Replace dash placeholders with the explicit 'N' category in selected columns."""

    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].replace("-", "N")
    return out


def normalize_yes_no_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert common yes/no codes into numeric category strings."""

    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].replace({"Y": "1", "N": "2"})
    return out


def replace_missing_placeholders(df: pd.DataFrame, missing_category: str = "missing") -> pd.DataFrame:
    """Replace string placeholders with a single missing category."""

    out = df.copy()
    out = out.replace({"-": np.nan, "": np.nan, "nan": np.nan, "<NA>": np.nan})
    return out.fillna(missing_category)


def impute_numeric_sentinels(
    df: pd.DataFrame,
    columns: Sequence[str],
    sentinel_values: Sequence[float],
    ignore_values_for_mean: Sequence[float] = (-1,),
) -> pd.DataFrame:
    """Replace sentinel values with the column mean calculated from valid observations."""

    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        numeric = pd.to_numeric(out[column], errors="coerce")
        numeric = numeric.mask(numeric.isin(sentinel_values), np.nan)
        valid = numeric[~numeric.isin(ignore_values_for_mean)]
        mean_value = valid.mean()
        out[column] = numeric.fillna(mean_value)
    return out


def bin_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    bins: Sequence[float],
    labels: Sequence[str],
) -> pd.DataFrame:
    """Categorize numeric columns according to fixed interval definitions."""

    out = df.copy()
    for column in columns:
        if column in out.columns:
            numeric = pd.to_numeric(out[column], errors="coerce")
            out[column] = pd.cut(numeric, bins=bins, labels=labels).astype("object")
    return out


def bin_vital_signs(df: pd.DataFrame, specs: Mapping[str, Mapping[str, Sequence]] | None = None) -> pd.DataFrame:
    """Apply all configured vital-sign binning rules."""

    out = df.copy()
    specs = specs or VITAL_BIN_SPECS
    for spec in specs.values():
        out = bin_columns(
            out,
            columns=spec["columns"],
            bins=spec["bins"],
            labels=spec["labels"],
        )
    return out


def map_gcs_to_category(value: object) -> str:
    """Map GCS-like values into broad severity categories."""

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or numeric < 0:
        return "missing"
    if numeric <= 8:
        return "severe"
    if numeric <= 12:
        return "moderate"
    return "mild"


def add_composite_index(df: pd.DataFrame, org_column: str, patient_column: str, output_column: str = "patient_id") -> pd.DataFrame:
    """Create a composite patient key from organization and patient-code columns."""

    out = df.copy()
    if org_column in out.columns and patient_column in out.columns:
        out[output_column] = out[org_column].astype(str) + out[patient_column].astype(str)
        out = out.set_index(output_column)
    return out


def one_hot_encode_features(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: Sequence[str] | None = None,
    missing_category: str = "missing",
) -> pd.DataFrame:
    """One-hot encode categorical variables and keep the target column unchanged."""

    out = df.copy()
    if categorical_columns is None:
        categorical_columns = [
            c for c in out.columns
            if c != target_column and (out[c].dtype == "object" or str(out[c].dtype) == "category")
        ]

    for column in categorical_columns:
        if column in out.columns:
            out[column] = out[column].fillna(missing_category).astype("object")

    out = pd.get_dummies(out, columns=[c for c in categorical_columns if c in out.columns], dummy_na=False)
    out = out.replace({True: 1, False: 0})
    return out


def drop_single_value_columns(df: pd.DataFrame, exclude: Sequence[str] = ()) -> pd.DataFrame:
    """Drop columns that have one or fewer unique values."""

    excluded = set(exclude)
    keep_columns = [c for c in df.columns if c in excluded or df[c].nunique(dropna=False) > 1]
    return df[keep_columns].copy()


def prepare_features(
    df: pd.DataFrame,
    target_column: str = "Survival",
    columns_to_drop: Sequence[str] = (),
    dash_to_no_columns: Sequence[str] = (),
    categorical_columns: Sequence[str] | None = None,
    missing_category: str = "missing",
    one_hot_encode: bool = True,
    remove_single_value_columns: bool = True,
) -> pd.DataFrame:
    """Run the public-safe feature-preparation sequence."""

    out = drop_existing_columns(df, columns_to_drop)
    out = replace_dash_with_no(out, dash_to_no_columns)
    out = replace_missing_placeholders(out, missing_category=missing_category)
    out = bin_vital_signs(out)

    if "trmsvgct" in out.columns:
        out["trmsvgct"] = out["trmsvgct"].apply(map_gcs_to_category)

    if one_hot_encode:
        out = one_hot_encode_features(
            out,
            target_column=target_column,
            categorical_columns=categorical_columns,
            missing_category=missing_category,
        )

    if remove_single_value_columns:
        out = drop_single_value_columns(out, exclude=[target_column])

    return out
