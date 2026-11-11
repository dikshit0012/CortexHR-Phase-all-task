"""Paths, seeds and the one place where the existing ``cortexhr`` package is located."""
from __future__ import annotations

import os
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[2]            # .../task 13
REPO_ROOT = TASK_ROOT.parent
REPORTS = TASK_ROOT / "reports"
ARTIFACTS = TASK_ROOT / "model_artifacts"

MODEL_VERSION = "attrition-v2"
RANDOM_STATE = 42          # same seed as Task 3 (folds) - never changed to improve results
N_SPLITS = 5
TARGET = "left_within_12m"

# --- locate the Task 2-4 package without copying it ------------------------------------------
_DEFAULT_SRC = REPO_ROOT.parent / "phase-1" / "task04" / "src"
CORTEXHR_SRC = Path(os.environ.get("CORTEXHR_ML_SRC", _DEFAULT_SRC))
if not (CORTEXHR_SRC / "cortexhr").is_dir():
    raise FileNotFoundError(
        f"Cannot find the Task 2-4 package 'cortexhr' in {CORTEXHR_SRC}. Set CORTEXHR_ML_SRC to the "
        "folder that contains it (e.g. 'phase-1/task04/src').")
if str(CORTEXHR_SRC) not in sys.path:
    sys.path.insert(0, str(CORTEXHR_SRC))

# --- the master workbook (not committed to git, see Task 1 README) ---------------------------
_DEFAULT_WORKBOOK = CORTEXHR_SRC.parent / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
WORKBOOK = Path(os.environ.get("CORTEXHR_WORKBOOK", _DEFAULT_WORKBOOK))
