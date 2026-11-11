"""Build the scoring table with Task 13's official function, validate it, and score all employees with attrition-v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .model_loader import mv


class ScoringDataError(ValueError):
    """The scoring data is not what Task 14 expects (reported, never silently repaired)."""


@dataclass
class ScoringInputs:
    X: pd.DataFrame              # the 33 model features, in training order (index = original row position)
    meta: pd.DataFrame           # employee_id, department_id, as_of_date (same index)
    missing_report: Dict[str, int]


def validate_ids(ids: pd.Series, expected: Optional[int] = config.EXPECTED_EMPLOYEES) -> None:
    if ids.isna().any() or (ids.astype("string").str.strip() == "").any():
        bad = list(ids[ids.isna() | (ids.astype("string").str.strip() == "")].index[:5])
        raise ScoringDataError(f"missing employee_id at rows {bad}")
    if ids.duplicated().any():
        raise ScoringDataError(f"duplicate employee_id values: {sorted(ids[ids.duplicated()].unique())[:5]}")
    if expected is not None and len(ids) != expected:
        raise ScoringDataError(f"expected exactly {expected} employees, found {len(ids)}")


def validate_features(X: pd.DataFrame, required: List[str]) -> Dict[str, int]:
    missing_cols = [c for c in required if c not in X.columns]
    extra_cols = [c for c in X.columns if c not in required]
    if missing_cols:
        raise ScoringDataError(f"missing required model feature column(s): {missing_cols}")
    if extra_cols:
        raise ScoringDataError(f"unexpected column(s) would be silently used: {extra_cols}")
    if list(X.columns) != list(required):
        raise ScoringDataError("feature order differs from training")
    return {c: int(n) for c, n in X.isna().sum().items() if n}


def build_scoring_inputs(loaded, workbook=None, expected: Optional[int] = config.EXPECTED_EMPLOYEES) -> ScoringInputs:
    path = workbook or mv.config.WORKBOOK
    raw = pd.read_excel(path, sheet_name=config.SCORING_SHEET)
    if "employee_id" not in raw.columns:
        raise ScoringDataError("scoring sheet has no employee_id column")
    validate_ids(raw["employee_id"], expected)
    X = mv.build_scoring_table(path)                       # Task 13 / Task 2: row rules, monthly features, safe-column selection
    if len(X) != len(raw) or not X.index.equals(raw.index):
        raise ScoringDataError("scoring table rows are not aligned with the scoring sheet (rows were dropped or reordered)")
    if "department_id" in raw.columns and not (X["department_id"].astype("string").values == raw["department_id"].astype("string").values).all():
        raise ScoringDataError("department_id differs between the scoring table and the sheet: rows are misaligned")
    missing = validate_features(X, loaded.required_columns)
    meta = pd.DataFrame({"employee_id": raw["employee_id"].astype(str).str.strip(), "department_id": raw["department_id"].astype(str),
                         "as_of_date": pd.to_datetime(raw["observation_date"]).dt.strftime("%Y-%m-%d")}, index=raw.index)
    return ScoringInputs(X=X, meta=meta, missing_report=missing)


def score_employees(inputs: ScoringInputs, loaded) -> pd.DataFrame:
    """attrition-v2 probability of leaving within 12 months (positive class), aligned to employee ids."""
    out = mv.predict_attrition(inputs.X, loaded)           # checks 0 <= p <= 1 and returns the model version
    p = out["attrition_probability_12m"].to_numpy(float)
    validate_probabilities(p)
    res = pd.DataFrame({"employee_id": inputs.meta["employee_id"].to_numpy(), "attrition_probability": p,
                        "model_version": out["model_version"].to_numpy()}, index=inputs.X.index)
    if len(res) != len(inputs.X) or res["employee_id"].duplicated().any():
        raise ScoringDataError("scores are not aligned one-to-one with employees")
    return res


def validate_probabilities(p) -> None:
    p = np.asarray(p, float)
    if p.size == 0 or np.isnan(p).any() or np.isinf(p).any():
        raise ScoringDataError("missing or non-finite attrition probabilities")
    if (p < 0).any() or (p > 1).any():
        raise ScoringDataError("attrition probabilities outside [0, 1]")
