"""Summarise the monthly attendance and payroll tables and join them to the attrition rows.

The monthly tables hold one row per employee per month. The attrition table holds one row per
employee per *observation date*. To avoid leaking the future, every attrition row only sees the
months that were COMPLETE on its own observation date:

    window = the ``window_months`` most recent completed months (minus an optional ``lag_months``)

For the real snapshot (observation_date = 30 Sep 2025) and the default 3-month window that is
July, August and September 2025. Months after the observation date are never used, and the
target is never read here.

Shape guarantee: the output has exactly as many rows as the input (checked, not assumed).
"""
from __future__ import annotations

import pandas as pd

from .paths import WORKBOOK

WINDOW_MONTHS = 3
KEY = "employee_id"

ATTENDANCE_SHEET = "attendance_monthly_features"
PAYROLL_SHEET = "payroll_monthly_features"
ID_COLUMNS = {ATTENDANCE_SHEET: "attendance_feature_id", PAYROLL_SHEET: "payroll_item_id"}


def monthly_feature_names(window_months: int = WINDOW_MONTHS) -> dict[str, list[str]]:
    """Names of the columns produced by :func:`add_monthly_features`, by role."""
    w = f"{window_months}m"
    return {
        # used as model inputs (see features.py for why)
        "model": [f"attendance_rate_change_{w}", f"avg_unpaid_absence_days_{w}", f"avg_variable_pay_{w}"],
        # kept for transparency / validation, deliberately NOT model inputs
        "reference": [f"avg_attendance_rate_{w}", "attendance_months_in_window", "payroll_months_in_window"],
    }


def load_monthly_tables(path=WORKBOOK) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two monthly sheets exactly as delivered."""
    return (pd.read_excel(path, sheet_name=ATTENDANCE_SHEET),
            pd.read_excel(path, sheet_name=PAYROLL_SHEET))


# ----------------------------------------------------------------------------------------------
# window logic
# ----------------------------------------------------------------------------------------------
def window_bounds(observation_date, window_months: int = WINDOW_MONTHS, lag_months: int = 0):
    """First-day-of-month timestamps (start, end), both inclusive, of the window for one date.

    A month only counts once it has finished: if ``observation_date`` is the last day of a month
    that month is included, otherwise the window ends with the previous month.
    ``lag_months`` shifts the window further back (a conservative option if the latest month is
    not available yet at the observation date).
    """
    if window_months < 1 or lag_months < 0:
        raise ValueError("window_months must be >= 1 and lag_months >= 0.")
    obs = pd.Timestamp(observation_date).normalize()
    period = obs.to_period("M")
    last_complete = period if obs == period.end_time.normalize() else period - 1
    end = (last_complete - lag_months).start_time
    start = (last_complete - lag_months - (window_months - 1)).start_time
    return start, end


# ----------------------------------------------------------------------------------------------
# data-quality guard
# ----------------------------------------------------------------------------------------------
def drop_duplicate_employee_months(df: pd.DataFrame, sheet: str) -> tuple[pd.DataFrame, int]:
    """Remove EXACT duplicate employee-month rows; refuse to guess when duplicates disagree.

    Rows are exact duplicates when every column except the row-id column is identical.
    Two rows with the same employee and month but different values are a real conflict, so we raise.
    """
    id_col = ID_COLUMNS[sheet]
    same_key = df.duplicated([KEY, "month"], keep=False)
    if not same_key.any():
        return df, 0
    content = [c for c in df.columns if c != id_col]
    exact = df.duplicated(content, keep="first")
    conflicts = df[same_key].drop_duplicates(content).duplicated([KEY, "month"], keep=False)
    if conflicts.any():
        bad = df[same_key].drop_duplicates(content).loc[conflicts, [KEY, "month"]].drop_duplicates()
        raise ValueError(
            f"{sheet}: {len(bad)} employee-month key(s) appear more than once with DIFFERENT values "
            f"(e.g. {bad.head(3).values.tolist()}). Resolve with the data owner; not guessing which to keep.")
    return df.loc[~exact].copy(), int(exact.sum())


# ----------------------------------------------------------------------------------------------
# aggregation + join
# ----------------------------------------------------------------------------------------------
def _rows_in_window(rows: pd.DataFrame, monthly: pd.DataFrame, window_months: int, lag_months: int) -> pd.DataFrame:
    """One-to-many join of attrition rows to their own window of monthly records (then filtered)."""
    bounds = {d: window_bounds(d, window_months, lag_months) for d in rows["observation_date"].unique()}
    rows = rows.assign(
        _start=rows["observation_date"].map(lambda d: bounds[d][0]),
        _end=rows["observation_date"].map(lambda d: bounds[d][1]))
    merged = rows.merge(monthly, on=KEY, how="inner")
    return merged[(merged["month"] >= merged["_start"]) & (merged["month"] <= merged["_end"])]


def add_monthly_features(attrition: pd.DataFrame, attendance: pd.DataFrame, payroll: pd.DataFrame,
                         window_months: int = WINDOW_MONTHS, lag_months: int = 0,
                         ) -> tuple[pd.DataFrame, dict]:
    """Return ``attrition`` plus point-in-time monthly features, and a diagnostics dict.

    Employees with no monthly records (e.g. every simulated HIST-xxxx row) get NaN, NOT zero:
    zero attendance or zero pay would be a false statement. Imputation happens later, inside the
    sklearn pipeline, fitted on training data only.
    """
    for col in (KEY, "observation_date"):
        if col not in attrition.columns:
            raise KeyError(f"attrition table needs a '{col}' column.")
    if attrition.index.has_duplicates:
        raise ValueError("attrition index must be unique; call reset_index(drop=True) first.")

    n_in = len(attrition)
    w = f"{window_months}m"

    att, n_dup_att = drop_duplicate_employee_months(attendance, ATTENDANCE_SHEET)
    pay, n_dup_pay = drop_duplicate_employee_months(payroll, PAYROLL_SHEET)

    rows = attrition[[KEY, "observation_date"]].copy()
    rows["_rid"] = attrition.index

    # --- attendance ----------------------------------------------------------------------
    a = _rows_in_window(rows, att[[KEY, "month", "attendance_rate", "unpaid_absence_days"]], window_months, lag_months)
    a = a.sort_values(["_rid", "month"])
    g = a.groupby("_rid")
    att_feat = pd.DataFrame({
        f"avg_attendance_rate_{w}": g["attendance_rate"].mean(),
        f"avg_unpaid_absence_days_{w}": g["unpaid_absence_days"].mean(),
        "attendance_months_in_window": g.size(),
    })
    # trend = latest month's attendance minus earliest month's, needs at least two months
    change = g["attendance_rate"].last() - g["attendance_rate"].first()
    att_feat[f"attendance_rate_change_{w}"] = change.where(att_feat["attendance_months_in_window"] >= 2)

    # --- payroll -------------------------------------------------------------------------
    p = _rows_in_window(rows, pay[[KEY, "month", "variable_pay"]], window_months, lag_months)
    gp = p.groupby("_rid")
    pay_feat = pd.DataFrame({f"avg_variable_pay_{w}": gp["variable_pay"].mean(),
                             "payroll_months_in_window": gp.size()})

    # --- join back: exactly one output row per input row ------------------------------------
    feats = att_feat.join(pay_feat, how="outer")
    out = attrition.copy()
    new_cols = [c for c in feats.columns]
    clash = set(new_cols) & set(out.columns)
    if clash:
        raise ValueError(f"attrition table already has columns {sorted(clash)}; drop them before joining.")
    out = out.join(feats, how="left", validate="one_to_one")
    if len(out) != n_in or out.index.has_duplicates:
        raise AssertionError(f"Join changed the row count: {n_in} -> {len(out)}. Aborting.")

    # months-in-window are counts: absent employees have 0 months of data (a true count, not a guess)
    for c in ("attendance_months_in_window", "payroll_months_in_window"):
        out[c] = out[c].fillna(0).astype(int)

    # --- point-in-time proof: no record used is later than its observation date ---------------
    latest_used = pd.concat([a["month"], p["month"]]).max() if len(a) or len(p) else pd.NaT
    obs_end_check = bool(((a["month"] + pd.offsets.MonthEnd(0)) <= a["observation_date"]).all()
                         and ((p["month"] + pd.offsets.MonthEnd(0)) <= p["observation_date"]).all())
    if lag_months == 0 and not obs_end_check:
        raise AssertionError("A monthly record used lies after its observation date (look-ahead).")

    diag = {
        "rows_in": n_in, "rows_out": len(out),
        "window_months": window_months, "lag_months": lag_months,
        "windows_used": {str(pd.Timestamp(d).date()): [str(b[0].date()), str(b[1].date())]
                         for d, b in list({d: window_bounds(d, window_months, lag_months)
                                           for d in rows["observation_date"].unique()}.items())[:5]},
        "duplicate_rows_removed": {ATTENDANCE_SHEET: n_dup_att, PAYROLL_SHEET: n_dup_pay},
        "rows_with_attendance_data": int((out["attendance_months_in_window"] > 0).sum()),
        "rows_without_attendance_data": int((out["attendance_months_in_window"] == 0).sum()),
        "rows_with_payroll_data": int((out["payroll_months_in_window"] > 0).sum()),
        "attendance_months_in_window_counts": out["attendance_months_in_window"].value_counts().sort_index().to_dict(),
        "payroll_months_in_window_counts": out["payroll_months_in_window"].value_counts().sort_index().to_dict(),
        "latest_month_used": None if pd.isna(latest_used) else str(pd.Timestamp(latest_used).date()),
        "no_look_ahead": obs_end_check,
    }
    return out, diag
