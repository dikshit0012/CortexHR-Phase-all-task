"""Cleaning, validation and series extraction for monthly_headcount_history."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .data_loader import DataError

log = logging.getLogger(__name__)

KEY = ["month", "department_id"]


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates to month-start, drop exact duplicate rows, reject conflicting duplicates and negative counts."""
    out = df.copy()
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    if out["month"].isna().any():
        raise DataError(f"{int(out['month'].isna().sum())} rows have an unparseable month")
    out["month"] = out["month"].dt.to_period("M").dt.to_timestamp()
    out["department_id"] = out["department_id"].astype(str).str.strip()
    before = len(out)
    out = out.drop_duplicates()
    if len(out) < before:
        log.warning("Dropped %d exact duplicate rows", before - len(out))
    conflicts = out.duplicated(KEY, keep=False)
    if conflicts.any():
        raise DataError(f"{int(conflicts.sum())} rows share (month, department_id) but hold different values")
    for col in ("headcount_end", "headcount_start", "hires", "exits"):
        if (out[col] < 0).any():
            raise DataError(f"Negative values found in {col}")
    return out.sort_values(KEY).reset_index(drop=True)


def validate_history(df: pd.DataFrame) -> dict:
    """Data-quality facts used by the EDA report. Nothing is modified."""
    full = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    missing_months = {
        dept: [m.strftime("%Y-%m") for m in full.difference(g["month"])] for dept, g in df.groupby("department_id")
    }
    flows = df.assign(
        identity_gap=df["headcount_start"] + df["hires"] - df["exits"] + df["transfers_in"] - df["transfers_out"] - df["headcount_end"]
    )
    company = df[df["department_id"] == config.COMPANY].sort_values("month")
    parts = df[df["department_id"] != config.COMPANY].groupby("month")[["headcount_end", "exits", "hires"]].sum()
    comp_idx = company.set_index("month")
    return {
        "rows": int(len(df)),
        "series": int(df["department_id"].nunique()),
        "first_month": df["month"].min().strftime("%Y-%m"),
        "last_month": df["month"].max().strftime("%Y-%m"),
        "n_months": int(len(full)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_keys": int(df.duplicated(KEY).sum()),
        "missing_months": {k: v for k, v in missing_months.items() if v},
        "flow_identity_violations": int((flows["identity_gap"] != 0).sum()),
        "start_end_continuity_violations": int((company["headcount_start"].iloc[1:].values != company["headcount_end"].iloc[:-1].values).sum()),
        "departments_sum_equals_company": bool((parts["headcount_end"] == comp_idx["headcount_end"]).all()),
    }


def get_series(df: pd.DataFrame, dept: str = config.COMPANY, col: str = "headcount_end") -> tuple[pd.Series, list[str]]:
    """Return a gap-free month-start series. Missing months are linearly interpolated and reported (never silent)."""
    g = df[df["department_id"] == dept]
    if g.empty:
        raise DataError(f"No rows for department '{dept}'")
    s = g.set_index("month")[col].sort_index().astype(float)
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    imputed = [m.strftime("%Y-%m") for m in full.difference(s.index)]
    s = s.reindex(full)
    if imputed:
        log.warning("%s/%s: interpolating %d missing months: %s", dept, col, len(imputed), imputed)
        s = s.interpolate(method="time")
    s.index.freq = "MS"
    s.name = col
    return s, imputed


def future_index(last: pd.Timestamp, horizon: int = config.HORIZON) -> pd.DatetimeIndex:
    return pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")


def forecast_departments(df: pd.DataFrame) -> list[str]:
    """Departments to forecast: the company total plus departments with a meaningful headcount (D00, a single person, is skipped
    only when the sample_forecast contract excludes it - see forecasting.py)."""
    return [config.COMPANY] + sorted(d for d in df["department_id"].unique() if d != config.COMPANY)
