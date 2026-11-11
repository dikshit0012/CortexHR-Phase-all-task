# Task 5 - Attendance anomaly detection

Task 5 detects unusual attendance records with an Isolation Forest and explicit business rules, then evaluates the results against the workbook's attendance labels. Its source, scripts, and tests are included; the master workbook is not.

## Setup

Use Python 3.10 or newer and install this task's dependencies from the repository root:

```powershell
python -m pip install -r phase-2\task05\requirements.txt
python -m pytest phase-2\task05\tests
```

The tests use synthetic data and do not require the workbook. To run the detector, request an authorized copy of `02_CortexHR_AI_ML_Dataset.xlsx` and set its path:

```powershell
$env:CORTEXHR_WORKBOOK = 'C:\approved-data\02_CortexHR_AI_ML_Dataset.xlsx'
python phase-2\task05\scripts\run_task5.py
```

Alternatively, place the workbook at `phase-2\task05\data\raw\02_CortexHR_AI_ML_Dataset.xlsx`, or pass `--workbook <path>` to `run_task5.py`. The workbook must contain `attendance_monthly_features`, `attendance_anomaly_labels` or `unlock_CX-05`, and `anomaly_baselines`.

`generate_phase1_report.py` writes PDF/DOCX files from report content embedded in the script; it does not rerun the analysis or read the workbook. Verify and update those reported results against authorized, current evidence before presenting them.

The runner writes record-level outputs under `phase-2\task05\reports`; those files are excluded from Git. Keep them within approved storage and do not share or commit them. The report generator requires both `python-docx` and `reportlab`, included in this task's requirements.
