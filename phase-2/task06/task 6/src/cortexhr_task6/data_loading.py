"""Task 6 - locating and reading the workbook (features + baseline). No label access here."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

WORKBOOK_NAME = "02_CortexHR_AI_ML_Dataset.xlsx"
FEATURE_SHEET = "payroll_monthly_features"
BASELINE_SHEET = "anomaly_baselines"
ENV_VAR = "CORTEXHR_WORKBOOK"


def find_workbook(explicit: Optional[str] = None) -> Path:
    """Locate the workbook from explicit/env paths, Task 6/5 data folders, or shared locations."""
    here = Path(__file__).resolve()
    task6_root, repo_root = here.parents[2], here.parents[4]
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(ENV_VAR):
        candidates.append(Path(os.environ[ENV_VAR]))
    candidates += [
        task6_root / "data" / "raw" / WORKBOOK_NAME,
        repo_root / "task05" / "data" / "raw" / WORKBOOK_NAME,
        repo_root / "Task 1" / "cortexhr-ml" / "data" / "raw" / WORKBOOK_NAME,
        repo_root.parent / "Phase 1 All task" / "task 5" / "data" / "raw" / WORKBOOK_NAME,
        repo_root.parent / "Phase 1 All task" / "Task 1" / "cortexhr-ml" / "data" / "raw" / WORKBOOK_NAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not find {WORKBOOK_NAME}. Put it in 'task06/task 6/data/raw/', or set {ENV_VAR}, "
        f"or pass --workbook.\nTried:\n{tried}"
    )


def load_payroll_features(xlsx_path) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, sheet_name=FEATURE_SHEET)


def load_baseline_row(xlsx_path) -> pd.Series:
    """Return the payroll row of the `anomaly_baselines` sheet."""
    base = pd.read_excel(xlsx_path, sheet_name=BASELINE_SHEET)
    hit = base[base["task"].str.contains("Payroll", case=False, na=False)]
    if hit.empty:
        raise ValueError(f"no payroll row found in sheet {BASELINE_SHEET}")
    return hit.iloc[0]
