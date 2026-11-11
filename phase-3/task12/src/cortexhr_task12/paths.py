"""Locations with environment overrides. Task 11 is imported from its own folder (never copied or modified)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

TASK12_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: Path) -> Path:
    return Path(os.environ[name]).expanduser() if os.environ.get(name) else default


TASK11_ROOT = _env("CORTEXHR_TASK11_ROOT", TASK12_ROOT.parent / "task11")
TASK11_SRC = TASK11_ROOT / "src"
TASK11_DATA = _env("CORTEXHR_TASK11_DATA", TASK11_ROOT / "data")             # resumes / job_descriptions / resume_job_labels
PAIRS_CSV = _env("CORTEXHR_TASK12_PAIRS", TASK12_ROOT / "data" / "resume_bias_pairs.csv")
PRIVATE_DIR = _env("CORTEXHR_TASK12_PRIVATE", TASK12_ROOT / "private")        # confidential unlock_CX-12 lives here (git-ignored)
PROTECTED_CSV = PRIVATE_DIR / "unlock_CX-12_resume_protected_attrs.csv"
REPORTS_DIR = _env("CORTEXHR_TASK12_REPORTS", TASK12_ROOT / "reports")


def ensure_task11_importable() -> None:
    if not (TASK11_SRC / "cortexhr_task11").exists():
        raise FileNotFoundError(f"Task 11 source not found at {TASK11_SRC}; set CORTEXHR_TASK11_ROOT")
    if str(TASK11_SRC) not in sys.path:
        sys.path.insert(0, str(TASK11_SRC))
