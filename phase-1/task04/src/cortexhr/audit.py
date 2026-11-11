"""Small, tested helpers used by the Task 1 audit (notebook + script).

Keeping the logic here (not buried in the notebook) means Task 2's pipeline can import
the very same rules.
"""
from __future__ import annotations

import pandas as pd

from .paths import WORKBOOK

TARGET = "left_within_12m"

# Columns that only exist AFTER the outcome is known, or are derived from the label.
# They must never be used as model inputs. (See notebook section 6 for the evidence.)
LEAKAGE_COLUMNS = [
    "exit_date",
    "resignation_date",
    "exit_reason",
    "exit_interview_done",
    "fnf_settlement_amount",
    "voluntary_left_within_12m",   # a sub-label of the target itself
]

# Not leakage in the strict sense, but not legitimate predictors either.
NON_FEATURE_COLUMNS = ["row_id", "employee_id", "source", "observation_date"]

LOCATION_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "madras": "Chennai",
    "chennai": "Chennai",
    "delhi ncr": "Delhi NCR",
    "delhi-ncr": "Delhi NCR",
}


def load_attrition(path=WORKBOOK) -> pd.DataFrame:
    """Read the master attrition sheet exactly as delivered (no cleaning)."""
    return pd.read_excel(path, sheet_name="attrition_dataset")


def iqr_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """True where a value lies beyond k x IQR outside the quartiles (Tukey fences)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def z_mask(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    """True where |z-score| exceeds the threshold (mean/std based)."""
    return ((s - s.mean()) / s.std()).abs() > threshold


def normalise_location(value):
    """Map messy city spellings to one canonical name; unknown values pass through trimmed."""
    if pd.isna(value):
        return value
    v = str(value).strip()
    return LOCATION_ALIASES.get(v.casefold(), v if v == v.title() else v.title())
