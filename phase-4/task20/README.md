# Task 20 - Final Audit Challenge

Task 20 contains the standalone input-completeness guard and experiment for the attrition scoring audit. It flags records missing critical monthly-derived features for human review; it does not impute values, change model scores, or modify the production API.

## Contents

- `completeness_guard.py`: pure record and batch validation helpers.
- `run_experiment.py`: optional audit experiment using the Task 13 environment and authorized local inputs.
- `tests/`: focused synthetic tests for guard behavior.

The audit, results, and model cards are documented under the repository's [`docs/`](../../docs/) folder.

## Run tests

From the repository root:

```powershell
python -m pytest phase-4\task20\tests -q
```

## Run the data-backed experiment

Use the Task 13 environment and approved local workbook/scoring CSV. These inputs and model artifacts are not committed.

```powershell
$env:CORTEXHR_ML_SRC = "phase-1/task04/src"
$env:CORTEXHR_WORKBOOK = "C:\approved-data\CortexHR_Dataset.xlsx"
python phase-4\task20\run_experiment.py "C:\approved-data\attrition_scoring_2026.csv"
```

The experiment writes aggregate metrics to `docs/final_challenge/results/`. Do not commit local source data, per-record predictions, or model artifacts.
