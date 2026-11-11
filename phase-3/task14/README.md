# Task 14 - Explainable attrition-risk report

Task 14 scores the current workforce with the Task 13 `attrition-v2` artifact and produces SHAP-based explanations. It is a decision-support demonstration, not an employment decision system. The model, workforce inputs, and generated individual-level reports are not included.

## Prerequisites

1. The shared Task 2-4 `cortexhr` package is included at `phase-1\task04\src`. Obtain the authorized master workbook through approved channels.
2. Set `CORTEXHR_ML_SRC` to an alternate `src` directory only if you are not using the included package. Set `CORTEXHR_WORKBOOK` to the authorized workbook file.
3. Use Task 13 to build `attrition-v2` from the same authorized workbook and package. Task 14 discovers the versioned artifact under `phase-3\task13\model_artifacts\attrition-v2` by default, or accepts `CORTEXHR_TASK13_ROOT`.
4. Preserve scikit-learn 1.8.x compatibility between the Task 13 artifact build and Task 14. Task 14's requirements constrain scikit-learn to `<1.9`; use the Task 13 requirements and recorded artifact metadata when building.

The authorized master workbook and Task 13 model artifact are external prerequisites; a clean clone cannot run Task 14 end to end by itself.

## Install, test, and run

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv-task14
.\.venv-task14\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r phase-3\task14\requirements.txt

$env:CORTEXHR_ML_SRC = (Resolve-Path phase-1\task04\src).Path
$env:CORTEXHR_WORKBOOK = 'C:\approved-data\02_CortexHR_AI_ML_Dataset.xlsx'
$env:CORTEXHR_TASK13_ROOT = (Resolve-Path phase-3\task13).Path

python -m pytest phase-3\task14\tests
python phase-3\task14\scripts\run_task14.py
```

The run takes several minutes when SHAP values are first computed. `--reports-dir` can direct output to an approved location. Task 14 expects the workbook sheets `attrition_scoring_2026`, `attendance_monthly_features`, and `payroll_monthly_features`; do not commit the workbook, model, or employee-level reports.
