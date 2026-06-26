"""Create public-safe cluster summary tables from a labeled private dataset."""

from __future__ import annotations

import argparse

from cpr_clustering.config import load_config
from cpr_clustering.io import read_table, save_table
from cpr_clustering.pipeline import summarize_clusters_public_safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize clusters.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--input", required=True, help="Path to a cluster-labeled private feature table.")
    parser.add_argument("--cluster-column", default="Cluster_Labels", help="Cluster label column name.")
    args = parser.parse_args()

    config = load_config(args.config)
    df = read_table(args.input, index_column=config.paths.index_column)
    summaries = summarize_clusters_public_safe(
        df,
        target_column=config.paths.target_column,
        cluster_column=args.cluster_column,
        redact_result_values=config.privacy.redact_result_values,
    )

    for name, table in summaries.items():
        output_path = config.paths.output_dir / f"{name}.csv"
        save_table(table, output_path, allow_private_output=True)
        print(f"Saved public-safe summary schema: {output_path}")


if __name__ == "__main__":
    main()
