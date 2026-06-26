"""Prepare the field-CPR cohort from private raw data.

This script does not ship with clinical data. Set paths in configs/config.example.yaml
before running locally.
"""

from __future__ import annotations

import argparse

from cpr_clustering.config import load_config
from cpr_clustering.io import save_table
from cpr_clustering.pipeline import prepare_cohort_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare field-CPR cohort.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    cohort = prepare_cohort_from_config(config)
    output_path = config.paths.output_dir / "cohort.csv"
    save_table(cohort, output_path, allow_private_output=config.privacy.save_private_outputs)

    if config.privacy.save_private_outputs:
        print(f"Saved cohort table: {output_path}")
    else:
        print("Cohort preparation completed. Private output writing is disabled.")


if __name__ == "__main__":
    main()
