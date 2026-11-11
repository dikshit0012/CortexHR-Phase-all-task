# Task 1-4: Attrition audit and modeling package

This package contains the shared `cortexhr` source used for the Task 1 data audit, Task 2 feature pipeline, Task 3 baselines, and Task 4 evaluation. It is included to make the package available to later tasks.

## Run tests

From the repository root in PowerShell:

```powershell
python -m pip install -r phase-1\task04\requirements.txt
python -m pytest phase-1\task04\tests -q
```

The local source package path for Task 13 and Task 14 is `phase-1\task04\src`. Set `CORTEXHR_ML_SRC` to its absolute path when running those tasks.

## Data and generated files

The master workbook, processed row-level scores, notebooks, and generated reports are intentionally not included. Obtain the workbook through an authorized channel and set `CORTEXHR_WORKBOOK` as documented in the task-specific instructions. Do not commit raw employee data, individual-level outputs, or model artifacts.

The source package was verified in the original data-enabled workspace with **110 tests passing**. In this repository, where the authorized workbook is intentionally omitted, **99 tests pass and 11 workbook-dependent tests are skipped**; 11 NumPy/joblib deprecation warnings are reported. The project requires Python packages listed in `requirements.txt`; serialized-model workflows should use the scikit-learn version documented by the consuming task.
