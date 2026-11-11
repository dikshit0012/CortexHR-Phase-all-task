"""Default locations (relative to the `task11` folder so they work on Windows and Linux) with environment overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

TASK11_ROOT = Path(__file__).resolve().parents[2]
ENV_DATA = "CORTEXHR_TASK11_DATA"
ENV_REPORTS = "CORTEXHR_TASK11_REPORTS"


def env_path(name: str, default: Path, environ: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if environ is None else environ
    return Path(env[name]).expanduser() if env.get(name) else default


DATA_DIR = env_path(ENV_DATA, TASK11_ROOT / "data")
REPORTS_DIR = env_path(ENV_REPORTS, TASK11_ROOT / "reports")
JOBS_CSV = DATA_DIR / "job_descriptions.csv"
RESUMES_CSV = DATA_DIR / "resumes.csv"
LABELS_CSV = DATA_DIR / "resume_job_labels.csv"
