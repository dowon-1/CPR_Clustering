"""Build model-ready features from the private field-CPR cohort."""

from __future__ import annotations

import argparse

from cpr_clustering.config import load_config
from cpr_clustering.io import save_table
from cpr_clustering.pipeline import prepare_features_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clustering features.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    features = prepare_features_from_config(config)
    output_path = config.paths.output_dir / "features.csv"
    save_table(features, output_path, allow_private_output=config.privacy.save_private_outputs)

    if config.privacy.save_private_outputs:
        print(f"Saved feature table: {output_path}")
    else:
        print("Feature preparation completed. Private output writing is disabled.")


if __name__ == "__main__":
    main()
