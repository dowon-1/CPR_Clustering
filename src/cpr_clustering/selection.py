"""Cohort selection and outcome-label creation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import CohortConfig


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def calculate_length_of_stay(
    df: pd.DataFrame,
    admission_date_column: str,
    transport_date_column: str,
    output_column: str = "LOS",
) -> pd.DataFrame:
    """Calculate hospital length of stay from two YYYYMMDD-style date columns."""

    out = df.copy()
    admission_date = pd.to_datetime(out[admission_date_column], format="%Y%m%d", errors="coerce")
    transport_date = pd.to_datetime(out[transport_date_column], format="%Y%m%d", errors="coerce")
    out[output_column] = (admission_date - transport_date).dt.days
    return out


def select_field_cpr_cohort(df: pd.DataFrame, config: CohortConfig) -> pd.DataFrame:
    """Apply reusable inclusion and exclusion rules for a field-CPR cohort."""

    out = df.copy()

    if config.field_cpr_column in out.columns:
        out = out[out[config.field_cpr_column].astype(str) == "Y"]

    if config.cpr_refusal_column in out.columns:
        out = out[out[config.cpr_refusal_column].astype(str) != "Y"]

    if config.dnr_column in out.columns:
        out = out[out[config.dnr_column].astype(str) != "Y"]

    if config.age_column in out.columns:
        out = out[_to_numeric_series(out[config.age_column]) >= config.minimum_age]

    if config.emergency_result_column in out.columns:
        out = out[~_to_numeric_series(out[config.emergency_result_column]).isin(config.excluded_emergency_result_codes)]

    if config.admission_result_column in out.columns:
        out = out[~_to_numeric_series(out[config.admission_result_column]).isin(config.excluded_admission_result_codes)]

    date_cols_exist = (
        config.admission_date_column in out.columns
        and config.transport_date_column in out.columns
        and config.admission_result_column in out.columns
    )
    if date_cols_exist:
        out = calculate_length_of_stay(
            out,
            admission_date_column=config.admission_date_column,
            transport_date_column=config.transport_date_column,
            output_column="LOS",
        )
        admission_result = _to_numeric_series(out[config.admission_result_column])
        short_stay = out["LOS"] < config.short_stay_exclusion_days
        out = out[~((admission_result == 3) & short_stay)]

    return out.reset_index(drop=True)


def create_survival_label(
    df: pd.DataFrame,
    config: CohortConfig,
    output_column: str = "Survival",
) -> pd.DataFrame:
    """Create a binary survival outcome label from emergency and admission result codes."""

    out = df.copy()
    admission_result = _to_numeric_series(out[config.admission_result_column])
    emergency_result = _to_numeric_series(out[config.emergency_result_column])

    conditions = [
        admission_result.isin(config.death_admission_result_codes),
        admission_result.isin(config.survival_admission_result_codes),
        emergency_result.isin(config.survival_emergency_result_codes),
    ]
    choices = [1, 0, 0]
    out[output_column] = np.select(conditions, choices, default=np.nan)
    out = out.dropna(subset=[output_column]).copy()
    out[output_column] = out[output_column].astype(int)
    return out.reset_index(drop=True)


def build_cohort(df: pd.DataFrame, config: CohortConfig, target_column: str = "Survival") -> pd.DataFrame:
    """Run cohort selection and outcome-label generation."""

    selected = select_field_cpr_cohort(df, config)
    return create_survival_label(selected, config, output_column=target_column)
