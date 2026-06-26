# Repository Structure

```text
configs/
  Example YAML configuration for private/local execution.

docs/
  Public documentation on privacy, structure, and usage.

sample/
  Schema-only sample files without patient rows.

scripts/
  Command-line entry points for each major workflow stage.

src/cpr_clustering/
  Reusable Python package.
```

## Module Overview

- `selection.py`: cohort selection and outcome labeling
- `cleaning.py`: missing-value handling, binning, and encoding
- `splitting.py`: train/validation splitting
- `distance.py`: mixed-type distance calculation
- `clustering.py`: hierarchical clustering and external assignment
- `statistics.py`: redacted cluster-level summaries
- `visualization.py`: optional private/local plots
- `pipeline.py`: high-level orchestration functions
