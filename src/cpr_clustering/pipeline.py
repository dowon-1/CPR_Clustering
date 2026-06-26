"""High-level public-safe pipeline functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cleaning import prepare_features
from .clustering import (
    assign_by_medoid_representatives,
    build_artifacts,
    evaluate_cluster_range,
    save_artifacts,
)
from .config import PipelineConfig
from .io import ensure_dir, read_table, save_table
from .selection import build_cohort
from .splitting import stratified_train_validation_split
from .statistics import cluster_outcome_summary, mann_whitney_cluster_tests


def prepare_cohort_from_config(config: PipelineConfig) -> pd.DataFrame:
    """Load raw data, select the field-CPR cohort, and create the target label."""

    df = read_table(config.paths.raw_input_path)
    cohort = build_cohort(df, config.cohort, target_column=config.paths.target_column)
    return cohort


def prepare_features_from_config(config: PipelineConfig, cohort: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prepare a model-ready feature table."""

    if cohort is None:
        cohort = prepare_cohort_from_config(config)

    features = prepare_features(
        cohort,
        target_column=config.paths.target_column,
        columns_to_drop=config.features.columns_to_drop,
        dash_to_no_columns=config.features.dash_to_no_columns,
        categorical_columns=config.features.categorical_columns or None,
        missing_category=config.features.missing_category,
        one_hot_encode=config.features.one_hot_encode,
        remove_single_value_columns=config.features.drop_single_value_columns,
    )
    return features


def fit_clustering_from_config(config: PipelineConfig) -> None:
    """Fit clustering artifacts without writing result values in public mode."""

    output_dir = ensure_dir(config.paths.output_dir)
    features = read_table(config.paths.feature_input_path or config.paths.raw_input_path, index_column=config.paths.index_column)

    train_df, valid_df = stratified_train_validation_split(
        features,
        target_column=config.paths.target_column,
        validation_size=config.split.validation_size,
        random_state=config.split.random_state,
        stratify=config.split.stratify,
    )

    artifacts = build_artifacts(
        df=train_df,
        target_column=config.paths.target_column,
        n_clusters=config.clustering.n_clusters,
        linkage_method=config.clustering.linkage_method,
        distance_metric=config.clustering.distance_metric,
    )

    eval_table = evaluate_cluster_range(
        train_df,
        linkage_matrix=artifacts.linkage_matrix,
        target_column=config.paths.target_column,
        cluster_min=config.clustering.cluster_range_min,
        cluster_max=config.clustering.cluster_range_max,
        redact_values=config.privacy.redact_result_values,
    )

    save_table(train_df, output_dir / "train_features.csv", allow_private_output=config.privacy.save_private_outputs)
    save_table(valid_df, output_dir / "valid_features.csv", allow_private_output=config.privacy.save_private_outputs)
    save_table(eval_table, output_dir / "cluster_range_evaluation.csv", allow_private_output=True)
    save_artifacts(artifacts, output_dir / "clustering_artifacts.joblib", allow_private_output=config.privacy.save_private_outputs)


def summarize_clusters_public_safe(
    df: pd.DataFrame,
    target_column: str,
    cluster_column: str = "Cluster_Labels",
    redact_result_values: bool = True,
) -> dict[str, pd.DataFrame]:
    """Create public-safe cluster summary tables."""

    return {
        "outcome_summary": cluster_outcome_summary(
            df,
            target_column=target_column,
            cluster_column=cluster_column,
            redact_values=redact_result_values,
        ),
        "mann_whitney_tests": mann_whitney_cluster_tests(
            df,
            cluster_column=cluster_column,
            excluded_columns=[target_column],
            redact_values=redact_result_values,
        ),
    }
