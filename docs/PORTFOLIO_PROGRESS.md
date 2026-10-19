# Personal portfolio development

Target repository: `naazzarov/BUS-BRA-Granular-Computing`.

This local project preserves the history and project specifications from
`Granular-Breast-Cancer-Team/BUS-BRA-Granular-Computing` at commit
`c01ab91863b9fdd25cae9ae859a7fd2b00f66623`. The original team checkout is unchanged.
The personal copy is published at
https://github.com/naazzarov/BUS-BRA-Granular-Computing on the `main` branch.
The baseline and shared infrastructure are available; the remaining methods and
real-data experiments are still in progress.

## Implemented

- Immutable CSV fingerprint verification and structural validation.
- The full 32-candidate baseline SVM search, five predefined patient-disjoint
  outer folds and three stratified patient-grouped inner folds.
- Training-only scaling, measured search/refit and prediction times, software
  versions, per-fold metrics and mean with sample standard deviation.
- A CLI that refuses missing or altered datasets and refuses to overwrite runs.

## Pending the original handoff and published references

The private ZIP, PDF, CSV, official comparison script and complete submission
templates are absent from this cloud workspace. Fuzzy-rough and GBFRS methods
still require implementation and source verification. No actual dataset
experiment or three-method comparison has been completed. Synthetic test runs
are software checks, not research results.

Do not describe this project as completed or cite synthetic scores on a CV.
Keep the team's attribution when presenting the completed personal extension.

## Development

```bash
uv sync --locked
uv run --locked pytest
uv run --locked busbra validate --data inputs/BUSBRA_features_ready.csv
uv run --locked busbra baseline --data inputs/BUSBRA_features_ready.csv --output private_outputs/baseline-run-001 --jobs 2
```

Only train when the unchanged team CSV passes validation. Output directories
contain private row-level artifacts. `internal_run.json` is an internal record;
official `best_params.csv`, `timings.csv`, and `run_info.json` must be exported
against the original ZIP templates once supplied.
