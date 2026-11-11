"""Task 7 - the official output format (sheet `sample_anomaly_results`).

OFFICIAL_COLUMNS is the exact column list and order of that sheet. The run script re-reads the sheet header from
the workbook and fails if it differs from this constant, so the format is checked against the real sample.
Internal helper columns never reach the official file.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

SAMPLE_SHEET = "sample_anomaly_results"
OFFICIAL_COLUMNS: List[str] = [
    "result_id", "source_dataset", "record_id", "employee_id", "period", "anomaly_score", "flagged",
    "reason_code", "reason_text", "money_at_risk", "model_version", "scored_at",
]
# The sample was scored in one batch at this timestamp. It is a run parameter (--scored-at) so reruns are
# byte-identical; pass the real run time when scoring for real.
DEFAULT_SCORED_AT = "2026-10-01 02:00:00"
MODEL_VERSIONS = {
    "payroll_monthly_features": "iforest+rules-payroll-task6",
    "attendance_monthly_features": "iforest+rules-attendance-task5",
}


def sample_columns(workbook: Union[str, Path]) -> List[str]:
    """Column names, in order, of the official sample sheet."""
    return list(pd.read_excel(workbook, sheet_name=SAMPLE_SHEET, nrows=0).columns)


def validate_official_schema(df: pd.DataFrame, expected: List[str] = None) -> None:
    """Raise unless the columns match exactly, in the same order."""
    expected = expected or OFFICIAL_COLUMNS
    cols = list(df.columns)
    if cols != expected:
        missing = [c for c in expected if c not in cols]
        extra = [c for c in cols if c not in expected]
        raise ValueError(f"official schema mismatch. expected {expected}; got {cols}; missing {missing}; extra {extra}; "
                         f"order_ok={[c for c in cols if c in expected] == [c for c in expected if c in cols]}")


def validate_official_values(df: pd.DataFrame) -> List[str]:
    """Value-level checks of the official file; returns a list of problems (empty = fine)."""
    problems: List[str] = []
    if df["result_id"].duplicated().any():
        problems.append("duplicate result_id")
    if not df["result_id"].astype(str).str.fullmatch(r"AN-\d{4,}").all():
        problems.append("result_id does not look like AN-0001")
    if df["record_id"].duplicated().any():
        problems.append("a record appears more than once")
    if not (df["flagged"] == "Y").all():
        problems.append("flagged must be 'Y' for every row of this file")
    if df["reason_code"].isna().any() or (df["reason_code"].astype(str).str.strip() == "").any():
        problems.append("empty reason_code")
    if df["reason_text"].isna().any() or (df["reason_text"].astype(str).str.strip() == "").any():
        problems.append("empty reason_text")
    score = pd.to_numeric(df["anomaly_score"], errors="coerce")
    if score.isna().any() or ((score < 0) | (score > 1)).any():
        problems.append("anomaly_score outside 0..1")
    money = pd.to_numeric(df["money_at_risk"], errors="coerce")
    if (money < 0).any() or np.isinf(money).any():
        problems.append("money_at_risk negative or infinite")
    if not df["period"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        problems.append("period is not YYYY-MM-DD")
    if not df["scored_at"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all():
        problems.append("scored_at is not YYYY-MM-DD HH:MM:SS")
    return problems


def to_official(ranked: pd.DataFrame, scored_at: str = DEFAULT_SCORED_AT) -> pd.DataFrame:
    """Ranked internal frame -> official frame (exact columns and order). result_id follows the ranking."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", scored_at):
        raise ValueError("scored_at must look like 'YYYY-MM-DD HH:MM:SS'")
    out = pd.DataFrame({
        "result_id": [f"AN-{i:04d}" for i in range(1, len(ranked) + 1)],
        "source_dataset": ranked["source_dataset"].to_numpy(),
        "record_id": ranked["record_id"].to_numpy(),
        "employee_id": ranked["employee_id"].to_numpy(),
        "period": pd.to_datetime(ranked["period"]).dt.strftime("%Y-%m-%d").to_numpy(),
        "anomaly_score": ranked["anomaly_score"].astype(float).round(3).to_numpy(),
        "flagged": "Y",
        "reason_code": ranked["primary_reason_code"].to_numpy(),
        "reason_text": ranked["reason_text"].to_numpy(),
        "money_at_risk": ranked["money_at_risk"].astype(float).round(2).to_numpy(),
        "model_version": ranked["source_dataset"].map(MODEL_VERSIONS).to_numpy(),
        "scored_at": scored_at,
    })
    validate_official_schema(out)
    return out
