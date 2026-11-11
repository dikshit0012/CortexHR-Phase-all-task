"""Task 6 - payroll feature engineering (ratios, history and role-reference features).

Nothing in this module reads or accepts ground-truth labels. Labels are only
touched in `payroll_evaluation.py`, after predictions have been produced.

Design rules:
  * Division by zero never produces a made-up number: the ratio is NaN. The model
    stage (not this module) decides how NaN is imputed, and says so.
  * History features use PRIOR months of the same employee only (no look-ahead).
  * The role reference (median fixed gross per role) is a separate fit/transform
    object, so it can be fitted on a reference window and re-used.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ID_COL = "payroll_item_id"
EMPLOYEE_COL = "employee_id"
MONTH_COL = "month"
ROLE_COL = "role_id"

MONEY_COLUMNS = [
    "monthly_fixed_gross", "overtime_pay", "variable_pay", "gross_pay",
    "pf_employee", "esi_employee", "professional_tax", "tds", "other_deductions",
    "total_deductions", "net_pay",
]
NUMERIC_COLUMNS = MONEY_COLUMNS + ["payable_days", "lop_days", "overtime_hours_paid", "job_level"]

REQUIRED_COLUMNS = [ID_COL, "run_id", EMPLOYEE_COL, MONTH_COL, ROLE_COL,
                    "overtime_eligible"] + NUMERIC_COLUMNS

# Columns that must NEVER be a model input: identifiers (the item id even carries a
# "PI-D" prefix that reveals double payments) and anything label-derived.
FORBIDDEN_MODEL_COLUMNS = [
    ID_COL, "run_id", EMPLOYEE_COL, MONTH_COL, ROLE_COL, "department_id",
    "anomaly_type", "field_changed", "original_value", "injection_note",
    "is_injected", "error_type",
]

# name -> (formula in words, why it helps)
FEATURE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "gross_to_fixed_gross": {
        "formula": "gross_pay / monthly_fixed_gross",
        "why": "Actual gross vs the employee's fixed entitlement. Zero pay (0) and wrong-decimal pay (~10x) sit far from the normal 0.3-3 range.",
    },
    "deduction_ratio": {
        "formula": "total_deductions / gross_pay",
        "why": "Share of gross taken in deductions. Excess tax pushes it above 1; a missing deduction lowers it. NaN when gross_pay is 0.",
    },
    "net_pay_ratio": {
        "formula": "net_pay / gross_pay",
        "why": "Take-home share of gross. Negative or tiny values signal over-deduction. NaN when gross_pay is 0.",
    },
    "gross_vs_expected": {
        "formula": "gross_pay / (monthly_fixed_gross * payable_days / days_in_month + overtime_pay + variable_pay)",
        "why": "Checks the data-dictionary identity 'gross = earned fixed + overtime + variable'. Should be ~1.0; a shifted decimal breaks it. NaN when the expected gross is 0.",
    },
    "fixed_gross_vs_role_median": {
        "formula": "monthly_fixed_gross / median monthly_fixed_gross for the same role_id in the previous calendar month",
        "why": "Pay relative to the prior month's role norm. A salary far above its role is a role/grade mismatch. NaN when no prior-month role reference exists.",
    },
    "fixed_gross_change_vs_prev": {
        "formula": "monthly_fixed_gross / the employee's previous-month monthly_fixed_gross",
        "why": "Month-on-month change in the fixed salary. A drop suggests payment on a stale (pre-revision) salary; a large jump suggests a wrong grade. NaN for an employee's first month.",
    },
    "overtime_hours_paid": {
        "formula": "overtime_hours_paid (raw)",
        "why": "Overtime hours are capped at 50 by policy; very large values indicate an overtime spike.",
    },
    "overtime_pay_share": {
        "formula": "overtime_pay / gross_pay",
        "why": "Overtime as a share of gross. Normal is near 0; a spike dominates the payslip. NaN when gross_pay is 0.",
    },
    "pf_vs_prev_month": {
        "formula": "pf_employee / the employee's previous-month pf_employee (only when the previous PF > 0)",
        "why": "PF is stable month to month; a drop to 0 means provident fund was not deducted. NaN when there is no previous month or previous PF was 0 (PF-exempt employees).",
    },
}

MODEL_FEATURES: List[str] = list(FEATURE_DEFINITIONS)


# --------------------------------------------------------------------------
# Safe maths
# --------------------------------------------------------------------------
def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, with NaN (not 0, not inf) wherever the result is undefined.

    Undefined = denominator is 0, either side is missing, or the result is not finite.
    """
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = num / den.where(den != 0)          # den == 0 -> NaN denominator -> NaN result
    return out.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_payroll_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns / integrity and return a cleaned copy.

    Invalid numbers become NaN (they are NOT silently replaced); infinities become NaN.
    Row order is preserved so outputs stay aligned with input records.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"payroll data is missing required columns: {missing}")
    if df.empty:
        raise ValueError("payroll data is empty")
    out = df.copy()
    key_columns = [ID_COL, "run_id", EMPLOYEE_COL, MONTH_COL, ROLE_COL]
    missing_keys = [c for c in key_columns if out[c].isna().any()]
    blank_keys = [
        c for c in key_columns if c != MONTH_COL
        and out[c].astype("string").str.strip().eq("").any()
    ]
    if missing_keys or blank_keys:
        raise ValueError(f"payroll data has missing or blank key values: {sorted(set(missing_keys + blank_keys))}")
    if out[ID_COL].duplicated().any():
        raise ValueError(f"duplicate values in {ID_COL}")
    out[MONTH_COL] = pd.to_datetime(out[MONTH_COL], errors="coerce")
    if out[MONTH_COL].isna().any():
        raise ValueError("some month values could not be parsed as dates")
    if out[MONTH_COL].dt.day.ne(1).any():
        raise ValueError("month values must be the first day of each calendar month")
    expected_run_ids = "PR-" + out[MONTH_COL].dt.strftime("%Y-%m")
    if out["run_id"].astype(str).ne(expected_run_ids).any():
        raise ValueError("run_id values must match their month as PR-YYYY-MM")
    for c in NUMERIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


# --------------------------------------------------------------------------
# Role reference (fit / transform so the reference window is explicit)
# --------------------------------------------------------------------------
class RoleReference:
    """Median monthly_fixed_gross per role_id in an explicitly supplied reference window."""

    def __init__(self) -> None:
        self.medians_: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "RoleReference":
        self.medians_ = df.groupby(ROLE_COL)["monthly_fixed_gross"].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.medians_ is None:
            raise RuntimeError("call fit() first")
        return df[ROLE_COL].map(self.medians_)


# --------------------------------------------------------------------------
# Feature building
# --------------------------------------------------------------------------
def month_period(df: pd.DataFrame) -> pd.Series:
    """Calendar month of each payslip as a Period (so 'previous month' is exact calendar arithmetic)."""
    return df[MONTH_COL].dt.to_period("M")


def add_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add history columns taken from the SAME employee's payslip in the previous CALENDAR month(s).

    prev_monthly_fixed_gross / prev_pf_employee : value one calendar month earlier
    prev2_monthly_fixed_gross                   : fixed gross two calendar months earlier

    * A missing month gives NaN ("no history"); the code never reaches back to an older row.
    * The result does not depend on the order of the input rows.
    * If an employee has several payslips in one month, the monthly_fixed_gross median and the
      largest pf_employee of that month are used, so repeats cannot become their own "previous month".
    """
    per_month = (
        df.assign(_period=month_period(df))
        .groupby([EMPLOYEE_COL, "_period"], as_index=False)
        .agg(fixed=("monthly_fixed_gross", "median"), pf=("pf_employee", "max"))
    )
    prev = per_month.assign(_period=per_month["_period"] + 1).rename(
        columns={"fixed": "prev_monthly_fixed_gross", "pf": "prev_pf_employee"})
    prev2 = per_month.assign(_period=per_month["_period"] + 2).rename(
        columns={"fixed": "prev2_monthly_fixed_gross"}).drop(columns="pf")
    key = df[[EMPLOYEE_COL]].assign(_period=month_period(df))
    out = df.copy()
    for table, cols in [(prev, ["prev_monthly_fixed_gross", "prev_pf_employee"]), (prev2, ["prev2_monthly_fixed_gross"])]:
        merged = key.merge(table, on=[EMPLOYEE_COL, "_period"], how="left")
        for c in cols:
            out[c] = merged[c].to_numpy()
    return out


def previous_month_role_median(df: pd.DataFrame) -> pd.Series:
    """Role median from the preceding calendar month, never the current or a future month."""
    period = month_period(df)
    by_role_month = (
        df.assign(_period=period)
        .groupby([ROLE_COL, "_period"])["monthly_fixed_gross"]
        .median()
    )
    previous_period = pd.MultiIndex.from_arrays([df[ROLE_COL], period - 1])
    return pd.Series(by_role_month.reindex(previous_period).to_numpy(), index=df.index)


def build_payroll_features(df: pd.DataFrame, role_reference: Optional[RoleReference] = None) -> pd.DataFrame:
    """Validate raw payroll rows and return them with the engineered ratio features.

    All original columns are kept (needed for rules, evaluation and reporting).
    """
    out = add_history_columns(validate_payroll_frame(df))
    days_in_month = out[MONTH_COL].dt.days_in_month
    earned_fixed = out["monthly_fixed_gross"] * out["payable_days"] / days_in_month
    out["expected_gross"] = earned_fixed + out["overtime_pay"] + out["variable_pay"]

    if role_reference is None:
        out["role_median_fixed_gross"] = previous_month_role_median(out)
    else:
        out["role_median_fixed_gross"] = role_reference.transform(out)

    out["gross_to_fixed_gross"] = safe_divide(out["gross_pay"], out["monthly_fixed_gross"])
    out["deduction_ratio"] = safe_divide(out["total_deductions"], out["gross_pay"])
    out["net_pay_ratio"] = safe_divide(out["net_pay"], out["gross_pay"])
    out["gross_vs_expected"] = safe_divide(out["gross_pay"], out["expected_gross"])
    out["fixed_gross_vs_role_median"] = safe_divide(out["monthly_fixed_gross"], out["role_median_fixed_gross"])
    out["fixed_gross_change_vs_prev"] = safe_divide(out["monthly_fixed_gross"], out["prev_monthly_fixed_gross"])
    out["overtime_pay_share"] = safe_divide(out["overtime_pay"], out["gross_pay"])
    # PF history is only meaningful where the previous PF was positive (PF-exempt staff stay NaN)
    prev_pf = out["prev_pf_employee"].where(out["prev_pf_employee"] > 0)
    out["pf_vs_prev_month"] = safe_divide(out["pf_employee"], prev_pf)
    return out
