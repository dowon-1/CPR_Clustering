# Data Privacy and Public-Release Notes

This public repository intentionally excludes clinical data and result values.

## Excluded Material

The following files must not be committed:

- Raw EMS or hospital data
- Patient identifiers
- Intermediate preprocessing tables
- Cluster-labeled patient files
- External-site validation files
- Statistical result tables with real numeric values
- Linkage matrices, medoids, centroids, and saved model artifacts
- Figures generated from private data

## Redaction Strategy

Public-facing summary functions support `redact_result_values=True`. In this mode, the table schema is preserved, but numeric results are replaced with `REDACTED`.

## Local Execution

For private/local execution, create a local configuration file that is not committed to Git. Set:

```yaml
privacy:
  redact_result_values: false
  save_private_outputs: true
  save_figures: true
```

Use this only in a secure environment approved for clinical data analysis.
