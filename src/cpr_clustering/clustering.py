"""Hierarchical clustering and cluster-label assignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, squareform
from sklearn.metrics import silhouette_samples

from .distance import gower_distance_matrix


@dataclass
class ClusteringArtifacts:
    """Reusable objects generated from a fitted clustering workflow."""

    linkage_matrix: np.ndarray
    feature_columns: List[str]
    train_labels: np.ndarray
    representatives: pd.DataFrame
    representative_labels: np.ndarray


def _feature_frame(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    return df.drop(columns=[target_column], errors="ignore").copy()


def fit_hierarchical_clustering(
    df: pd.DataFrame,
    target_column: str,
    n_clusters: int,
    linkage_method: str = "ward",
    distance_metric: str = "gower",
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit hierarchical clustering and return linkage matrix plus labels."""

    x = _feature_frame(df, target_column)

    if distance_metric == "gower":
        distance_matrix = gower_distance_matrix(x)
        condensed = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed, method=linkage_method)
    elif distance_metric == "euclidean":
        x_numeric = x.apply(pd.to_numeric, errors="coerce").fillna(0)
        linkage_matrix = linkage(x_numeric.to_numpy(), method=linkage_method)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    return linkage_matrix, labels


def evaluate_cluster_range(
    df: pd.DataFrame,
    linkage_matrix: np.ndarray,
    target_column: str,
    cluster_min: int = 2,
    cluster_max: int = 10,
    redact_values: bool = True,
) -> pd.DataFrame:
    """Evaluate candidate cluster counts using silhouette summaries.

    When redact_values is True, the returned table contains the method-level schema
    but hides numeric values for public release.
    """

    x = _feature_frame(df, target_column).apply(pd.to_numeric, errors="coerce").fillna(0)
    rows = []
    for n_clusters in range(cluster_min, cluster_max + 1):
        labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            mean_score = np.nan
            std_score = np.nan
        else:
            values = silhouette_samples(x, labels)
            mean_score = float(np.mean(values))
            std_score = float(np.std([np.mean(values[labels == label]) for label in np.unique(labels)]))

        rows.append({
            "n_clusters": n_clusters,
            "silhouette_mean": "REDACTED" if redact_values else mean_score,
            "silhouette_cluster_mean_std": "REDACTED" if redact_values else std_score,
        })
    return pd.DataFrame(rows)


def select_medoid_representatives(x: pd.DataFrame, labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """Select one medoid-like representative row per cluster using Gower distance."""

    representative_indices = []
    representative_labels = []

    for label in sorted(np.unique(labels)):
        cluster_x = x.loc[labels == label]
        distances = gower_distance_matrix(cluster_x)
        medoid_position = int(np.argmin(distances.sum(axis=1)))
        representative_indices.append(cluster_x.index[medoid_position])
        representative_labels.append(label)

    representatives = x.loc[representative_indices].copy()
    return representatives, np.array(representative_labels)


def build_artifacts(
    df: pd.DataFrame,
    target_column: str,
    n_clusters: int,
    linkage_method: str = "ward",
    distance_metric: str = "gower",
) -> ClusteringArtifacts:
    """Fit clustering and create reusable assignment artifacts."""

    linkage_matrix, labels = fit_hierarchical_clustering(
        df=df,
        target_column=target_column,
        n_clusters=n_clusters,
        linkage_method=linkage_method,
        distance_metric=distance_metric,
    )
    x = _feature_frame(df, target_column)
    representatives, representative_labels = select_medoid_representatives(x, labels)
    return ClusteringArtifacts(
        linkage_matrix=linkage_matrix,
        feature_columns=list(x.columns),
        train_labels=labels,
        representatives=representatives,
        representative_labels=representative_labels,
    )


def add_cluster_labels(df: pd.DataFrame, labels: np.ndarray, output_column: str = "Cluster_Labels") -> pd.DataFrame:
    """Return a copy of a DataFrame with cluster labels appended."""

    out = df.copy()
    out[output_column] = labels
    return out


def assign_by_medoid_representatives(
    df: pd.DataFrame,
    artifacts: ClusteringArtifacts,
    target_column: str,
    output_column: str = "Cluster_Labels",
) -> pd.DataFrame:
    """Assign new cases to the nearest training-cluster representative."""

    x = _feature_frame(df, target_column).reindex(columns=artifacts.feature_columns, fill_value=0)
    distances = gower_distance_matrix(pd.concat([x, artifacts.representatives], axis=0))
    n_new = len(x)
    distance_to_reps = distances[:n_new, n_new:]
    label_positions = np.argmin(distance_to_reps, axis=1)
    labels = artifacts.representative_labels[label_positions]
    return add_cluster_labels(df, labels, output_column=output_column)


def assign_by_euclidean_centroids(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    train_labels: np.ndarray,
    target_column: str,
    output_column: str = "Cluster_Labels",
) -> pd.DataFrame:
    """Assign new cases to nearest numeric centroids from the training set."""

    train_x = _feature_frame(train_df, target_column).apply(pd.to_numeric, errors="coerce").fillna(0)
    new_x = _feature_frame(df, target_column).reindex(columns=train_x.columns, fill_value=0)
    new_x = new_x.apply(pd.to_numeric, errors="coerce").fillna(0)

    centroids = []
    centroid_labels = []
    for label in sorted(np.unique(train_labels)):
        centroids.append(train_x.loc[train_labels == label].mean(axis=0))
        centroid_labels.append(label)

    distances = cdist(new_x.to_numpy(), np.vstack(centroids), metric="euclidean")
    labels = np.array(centroid_labels)[np.argmin(distances, axis=1)]
    return add_cluster_labels(df, labels, output_column=output_column)


def save_artifacts(artifacts: ClusteringArtifacts, path: str | Path, allow_private_output: bool = False) -> None:
    """Save clustering artifacts only when private-output writing is explicitly enabled."""

    if not allow_private_output:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path)


def load_artifacts(path: str | Path) -> ClusteringArtifacts:
    """Load clustering artifacts saved with joblib."""

    return joblib.load(path)
