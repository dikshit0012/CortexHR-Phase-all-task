"""Paths (pathlib, env overrides), constants and the feature policy. Nothing here is a metric; every number is computed at run time."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: Path) -> Path:
    return Path(os.environ[name]).expanduser() if os.environ.get(name) else default


REPORTS = _env("CORTEXHR_TASK15_REPORTS", ROOT / "reports")
PRIVATE_REPORTS = REPORTS / "private"
MODELS = _env("CORTEXHR_TASK15_MODELS", ROOT / "reports" / "model_artifacts")
WORKBOOK = _env("CORTEXHR_WORKBOOK", ROOT / "data" / "CortexHR_Dataset.xlsx")      # master workbook (never committed)
SALARY_CSV = _env("CORTEXHR_SALARY_CSV", ROOT / "data" / "salary_training.csv")     # optional CSV export of the sheet
PRIVATE_DIR = _env("CORTEXHR_TASK15_PRIVATE", ROOT / "private")                      # where unlock_CX-15 is placed (git-ignored)
UNLOCK_NAMES = ("unlock_CX-15_protected_attributes.csv", "unlock_CX-15.csv", "protected_attributes.csv")

SHEET = "salary_training"
TARGET = "offered_annual_fixed"
ID_COL = "hire_id"
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
BAND_LOW_Q, BAND_HIGH_Q = 0.10, 0.90          # nominal 80% empirical band
MIN_GROUP = 30                                 # groups smaller than this are flagged as unreliable
MODEL_VERSION = "paybands-v1"
NOTE = ("Decision support only: an estimate learned from historical offers, which may contain past pay inequity. "
        "Final compensation must be reviewed by HR and an authorised manager; do not use as the sole basis for a pay decision.")

# ---------------------------------------------------------------- feature policy (every column of salary_training has a decision)
NUMERIC = ["job_level", "prior_experience_years", "market_index"]
CATEGORICAL = ["role_id", "department_id", "work_location", "highest_education"]
ALLOWED = NUMERIC + CATEGORICAL
AUDIT_ONLY = ["career_gap_months", "gender", "age_years", "age_band"]
POLICY = {
    "role_id": ("allowed", "the job itself: the main driver of the value of a role"),
    "job_level": ("allowed", "responsibility level (1 Associate ... 7 CEO)"),
    "department_id": ("allowed", "function / pay market of the role"),
    "prior_experience_years": ("allowed", "relevant experience before joining; a standard pay factor"),
    "market_index": ("allowed", "pay-inflation index (5.5% a year): converts hire-year pay to a common price level; not a person attribute"),
    "work_location": ("allowed (governed)", "cost-of-labour category; may proxy for region, so it is evaluated and can be removed (mitigation M2)"),
    "highest_education": ("allowed (governed)", "qualification; may proxy for socio-economic background, so it is evaluated and can be removed (mitigation M2)"),
    "career_gap_months": ("audit-only", "required for fairness testing; using it would build a past pay penalty into recommendations"),
    "gender": ("audit-only", "protected attribute from unlock_CX-15; fairness analysis only, never a predictive input"),
    "age_years": ("audit-only", "protected attribute from unlock_CX-15; fairness analysis only, never a predictive input"),
    "age_band": ("audit-only", "protected attribute from unlock_CX-15; fairness analysis only, never a predictive input"),
    "person_ref": ("audit-only", "unlock_CX-15 join key; used only to link authorized audit labels and never a predictive input"),
    "record_type": ("audit-only", "unlock_CX-15 provenance metadata; never a predictive input"),
    "band_min_2026": ("excluded", "2026 role band: circular (target / (band_mid x market_index) averages 0.99, log-correlation 0.99); it is the pay structure being recommended, and role_id already carries the role information"),
    "band_mid_2026": ("excluded", "as above"),
    "band_max_2026": ("excluded", "as above"),
    "hire_date": ("excluded", "date of joining; replaced by market_index (avoids a duplicate time trend and date-derived age signals)"),
    "hire_year": ("excluded", "redundant with market_index"),
    "skills_count": ("excluded", "documented as a noise variable; no job-relevant meaning"),
    "source": ("excluded", "data provenance (real vs extension rows), not a job attribute"),
    "hire_id": ("excluded", "identifier"),
    TARGET: ("target", "annual fixed pay offered at hire (INR, nominal)"),
}
SENSITIVE_NEVER_FEATURES = {"gender", "age_years", "age_band", "career_gap_months", "candidate_name", "dob", "religion", "caste", "marital_status", "person_ref", "record_type", "hire_id"}
