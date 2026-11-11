# CortexHR Workforce Analytics

This repository consolidates the CortexHR project in one place for review. It contains completed work for Phases 1-3, organized by phase and task. Phase 4 will be added when it is released and completed.

The project is a collection of separate experiments and services, not one install-and-run application. Follow each task's README for its Python version, dependencies, approved input data, environment variables, and run commands. Some tasks depend on local data or model artifacts that are intentionally not included.

## Repository layout

| Phase | Tasks | Work |
|---|---|---|
| [Phase 1](phase-1/) | Tasks 1-4 | Attrition data audit, feature pipeline, baselines, and shared modeling package |
| [Phase 2](phase-2/) | Tasks 5-10 | Attendance/payroll anomaly detection, explanations, API, FAQ classifier, and HR integration |
| [Phase 3](phase-3/) | Tasks 11-15 | Resume ranking and fairness, attrition model and explanations, and fair-pay recommendations |

Each phase contains numbered task folders (`task01`, `task02`, etc.). The Task 2-4 shared `cortexhr` package is in `phase-1/task04/src`; Tasks 13-14 use it from there.

## Set up and run

Use a separate virtual environment for each task because dependency versions differ. For example, from the repository root in Windows PowerShell:

```powershell
python -m venv .venv-task05
.\.venv-task05\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r phase-2\task05\requirements.txt
python -m pytest phase-2\task05\tests
```

The Task 5 tests use synthetic fixtures. Running its real-data pipeline requires an authorized local workbook:

```powershell
$env:CORTEXHR_WORKBOOK = 'C:\approved-data\02_CortexHR_AI_ML_Dataset.xlsx'
python phase-2\task05\scripts\run_task5.py
```

Other tasks may require additional data or locally generated model artifacts. Read their task-specific documentation before running them.

## Data, artifacts, and privacy

Do not commit workbooks, candidate or employee data, protected-attribute files, credentials, `.env` files, local models, or person-level outputs. Obtain required data through approved channels, keep it in access-controlled local paths, and configure the task-specific environment variables documented by each project. Ignore rules are not a substitute for checking Git status before pushing.

The programs are for research and decision support. They are not production-ready HR systems and must not be used as the sole basis for employment, compensation, or other consequential decisions. Use only approved data and follow applicable privacy and retention requirements.

## Verification

Run synthetic tests first. Data-backed tests and full pipelines require the approved inputs described in each task README. Existing reports may not be reproducible from a clean clone because some inputs and generated artifacts are intentionally omitted.

