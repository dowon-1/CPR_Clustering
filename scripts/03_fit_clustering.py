"""Fit hierarchical clustering on private/local feature data."""

from __future__ import annotations

import argparse

from cpr_clustering.config import load_config
from cpr_clustering.pipeline import fit_clustering_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit field-CPR clustering model.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    fit_clustering_from_config(config)
    print("Clustering workflow completed. Public-safe summaries do not contain result values.")


if __name__ == "__main__":
    main()
