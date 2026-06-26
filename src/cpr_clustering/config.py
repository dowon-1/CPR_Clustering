"""Configuration helpers for the field CPR clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PathConfig:
    """Input and output paths used by the pipeline."""

    raw_input_path: Path
    feature_input_path: Optional[Path] = None
    output_dir: Path = Path("outputs")
    index_column: str = "patient_id"
    target_column: str = "Survival"


@dataclass
class CohortConfig:
    """Column names and codes used to define the field-CPR cohort."""

    field_cpr_column: str = "trmspt31"
    cpr_refusal_column: str = "trmspt32"
    dnr_column: str = "trmspt33"
    age_column: str = "ptmibdrt"
    emergency_result_column: str = "ptmiemrt"
    admission_result_column: str = "ptmidcrt"
    admission_date_column: str = "ptmidcdt"
    transport_date_column: str = "trmsindt"
    minimum_age: int = 5
    excluded_emergency_result_codes: List[int] = field(default_factory=lambda: [
        11, 12, 13, 14, 15, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 48, 88
    ])
    excluded_admission_result_codes: List[int] = field(default_factory=lambda: [2, 5, 8])
    death_admission_result_codes: List[int] = field(default_factory=lambda: [1, 3])
    survival_admission_result_codes: List[int] = field(default_factory=lambda: [4, 6])
    survival_emergency_result_codes: List[int] = field(default_factory=lambda: [41, 42, 43, 44, 45])
    short_stay_exclusion_days: int = 8


@dataclass
class FeatureConfig:
    """Feature engineering options."""

    columns_to_drop: List[str] = field(default_factory=list)
    dash_to_no_columns: List[str] = field(default_factory=list)
    additional_drop_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    missing_category: str = "missing"
    one_hot_encode: bool = True
    drop_single_value_columns: bool = True


@dataclass
class SplitConfig:
    """Train and validation split options."""

    validation_size: float = 0.24
    random_state: int = 42
    stratify: bool = True


@dataclass
class ClusteringConfig:
    """Hierarchical clustering options."""

    n_clusters: int = 4
    cluster_range_min: int = 2
    cluster_range_max: int = 10
    linkage_method: str = "ward"
    distance_metric: str = "gower"
    representative_method: str = "medoid"


@dataclass
class PrivacyConfig:
    """Public-release safety options."""

    redact_result_values: bool = True
    save_private_outputs: bool = False
    save_figures: bool = False
    output_cluster_labels: bool = False


@dataclass
class PipelineConfig:
    """Top-level configuration object."""

    paths: PathConfig
    cohort: CohortConfig = field(default_factory=CohortConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


def _to_path_config(data: Dict[str, Any]) -> PathConfig:
    return PathConfig(
        raw_input_path=Path(data["raw_input_path"]),
        feature_input_path=Path(data["feature_input_path"]) if data.get("feature_input_path") else None,
        output_dir=Path(data.get("output_dir", "outputs")),
        index_column=data.get("index_column", "patient_id"),
        target_column=data.get("target_column", "Survival"),
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return PipelineConfig(
        paths=_to_path_config(data.get("paths", {})),
        cohort=CohortConfig(**data.get("cohort", {})),
        features=FeatureConfig(**data.get("features", {})),
        split=SplitConfig(**data.get("split", {})),
        clustering=ClusteringConfig(**data.get("clustering", {})),
        privacy=PrivacyConfig(**data.get("privacy", {})),
    )
