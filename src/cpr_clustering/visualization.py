"""Optional plotting utilities for private/local analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_silhouette(
    x: pd.DataFrame,
    linkage_matrix: np.ndarray,
    n_clusters: int,
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Create a silhouette plot for a selected number of clusters."""

    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    x_numeric = x.apply(pd.to_numeric, errors="coerce").fillna(0)

    if len(np.unique(labels)) < 2:
        return

    values = silhouette_samples(x_numeric, labels)
    average = silhouette_score(x_numeric, labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for cluster_id in sorted(np.unique(labels)):
        cluster_values = values[labels == cluster_id]
        cluster_values.sort()
        y_upper = y_lower + len(cluster_values)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values, alpha=0.7)
        ax.text(0.01, y_lower + 0.5 * len(cluster_values), str(cluster_id))
        y_lower = y_upper + 10

    ax.axvline(x=average, linestyle="--", linewidth=2)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster label")
    ax.set_yticks([])
    ax.set_title("Silhouette analysis")
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_dendrogram_heatmap(
    x: pd.DataFrame,
    linkage_matrix: np.ndarray,
    n_clusters: int,
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Create a dendrogram and heatmap figure for private/local review."""

    fig, (ax_dendro, ax_heatmap) = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        gridspec_kw={"height_ratios": [1, 3]},
    )

    threshold = linkage_matrix[-(n_clusters - 1), 2] if n_clusters > 1 else None
    dendrogram(
        linkage_matrix,
        ax=ax_dendro,
        orientation="top",
        color_threshold=threshold,
        above_threshold_color="grey",
    )
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])

    sorted_index = dendrogram(linkage_matrix, no_plot=True)["leaves"]
    x_numeric = x.apply(pd.to_numeric, errors="coerce").fillna(0)
    image = ax_heatmap.imshow(x_numeric.iloc[sorted_index].T, aspect="auto", interpolation="nearest")
    ax_heatmap.set_xlabel("Individual encounters")
    ax_heatmap.set_ylabel("Variables")
    ax_heatmap.set_xticks([])
    fig.colorbar(image, ax=ax_heatmap, fraction=0.02, pad=0.02)
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
