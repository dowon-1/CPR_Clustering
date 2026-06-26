# Field CPR Patient Clustering Pipeline

![clinical data](https://img.shields.io/badge/clinical_data-not_included-lightgrey)
![result values](https://img.shields.io/badge/result_values-redacted-lightgrey)
![status](https://img.shields.io/badge/status-research_code-blue)

This repository contains a public-safe Python implementation of a clustering workflow for patients who received CPR at the accident or prehospital scene.

The original exploratory analysis was developed in Jupyter notebooks. This public version reorganizes the workflow into reusable Python modules and scripts. Clinical data, hospital-specific file names, patient-level outputs, cluster assignments, p-values, mortality rates, figures, and trained clustering artifacts are not included.

## Repository Scope

The code supports the following local/private workflow:

1. Select a field-CPR cohort from private EMS or hospital-linked data.
2. Create a binary outcome label from emergency and admission result codes.
3. Clean variables and standardize missing-value handling.
4. Categorize vital signs and selected clinical variables.
5. One-hot encode categorical features.
6. Split data into train and validation sets.
7. Fit hierarchical clustering using mixed-type distance calculation.
8. Assign validation or external-site cases using training-cluster representatives.
9. Generate cluster-level summary tables with optional public redaction.

## Public Data Policy

The following are intentionally excluded:

- Raw clinical tables
- Patient identifiers
- Hospital-specific output files
- Cluster labels for real patients
- Cluster-size counts
- Mortality or survival rates
- P-values and feature-summary values
- Linkage matrices and trained artifacts
- Figures generated from private data

Only schema templates, simplified artificial sample rows, and code are provided.

## Repository Structure

```text
CPR_Field_CPR_Clustering/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   └── config.example.yaml
├── docs/
│   ├── data_privacy.md
│   └── repository_structure.md
├── sample/
│   └── sample_input_schema.csv
├── scripts/
│   ├── 01_prepare_cohort.py
│   ├── 02_build_features.py
│   ├── 03_fit_clustering.py
│   └── 04_summarize_clusters.py
└── src/
    └── cpr_clustering/
        ├── __init__.py
        ├── cleaning.py
        ├── clustering.py
        ├── config.py
        ├── distance.py
        ├── io.py
        ├── pipeline.py
        ├── selection.py
        ├── splitting.py
        ├── statistics.py
        └── visualization.py
```

## Main Components

### Cohort Selection

`src/cpr_clustering/selection.py`

This module defines reusable rules for selecting field-CPR patients and creating the target label. The rules are configurable through `configs/config.example.yaml`.

### Feature Cleaning

`src/cpr_clustering/cleaning.py`

This module includes utilities for dropping unused variables, handling missing placeholders, binning vital signs, mapping GCS-like values, creating patient keys, one-hot encoding, and removing single-value columns.

### Clustering

`src/cpr_clustering/clustering.py`

This module fits hierarchical clustering and creates reusable artifacts. It supports medoid-style representative selection and assignment of validation or external-site cases to the nearest representative cluster.

### Statistical Summaries

`src/cpr_clustering/statistics.py`

This module creates cluster-level summaries and Mann-Whitney U test tables. Public mode redacts numeric results by default.

### Visualization

`src/cpr_clustering/visualization.py`

This module contains optional plotting functions for private/local review. Figures should not be committed to the public repository.

## Installation

```bash
conda create -n cpr_cluster python=3.9
conda activate cpr_cluster
pip install -r requirements.txt
pip install -e .
```


## Sample Data

The `sample/` directory contains only simplified artificial rows. These files are intended to show the expected column format and do not contain real patient data, real hospital outputs, real cluster labels, or real result values.

- `sample_input_schema.csv`: minimal raw/cohort-level schema for cohort-selection examples.
- `sample_feature_input.csv`: minimal feature-level table similar to a one-hot encoded clustering input.

## Quick Start

Copy the example configuration and replace placeholder paths with private local paths.

```bash
cp configs/config.example.yaml configs/config.local.yaml
```

Prepare the cohort:

```bash
python scripts/01_prepare_cohort.py --config configs/config.local.yaml
```

Build features:

```bash
python scripts/02_build_features.py --config configs/config.local.yaml
```

Fit clustering:

```bash
python scripts/03_fit_clustering.py --config configs/config.local.yaml
```

Create public-safe summaries from a private cluster-labeled table:

```bash
python scripts/04_summarize_clusters.py \
  --config configs/config.local.yaml \
  --input <PRIVATE_OUTPUT_DIR>/cluster_labeled_features.csv
```

By default, `privacy.redact_result_values` is set to `true`, so public summary tables do not expose numeric result values.

## Notes for Public Release

Before pushing to GitHub, verify that no private files are staged:

```bash
git status
```

Also search for private terms and local paths:

```bash
grep -R "<PRIVATE_LOCAL_PATH>\|patient-level\|hospital_name\|Cluster_Labels" .
```

The term `Cluster_Labels` appears in source code as a generic column name. Real patient-level cluster labels must not be committed.

## Disclaimer

This repository is for research documentation and reproducibility of analysis code only. It is not a clinical decision-support system and must not be used for real-world medical decision-making.
