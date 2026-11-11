"""Task 7 - EVALUATION-ONLY checks against the payroll labels. This is the only Task 7 module that reads labels.

It runs AFTER the official output has been written and never feeds anything back into reason codes,
texts, amounts or ranking. It answers two questions about the payroll flags:
  1. Do the reason codes agree with the kind of error that was injected? (reason code x injected type)
  2. Are the money estimates near the injected change? (estimate vs a reference amount rebuilt from the label)
Labels are `payroll_anomaly_labels`, the substitute for the missing `unlock_CX-06` sheet, so results are provisional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_ID, LABEL_TYPE = "payroll_item_id", "anomaly_type"


def reason_vs_label(official: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Counts of payroll flags by reason code and injected error type (NO_LABEL = flag on an unlabelled payslip)."""
    pay = official[official["source_dataset"] == "payroll_monthly_features"]
    lab = labels.set_index(LABEL_ID)[LABEL_TYPE]
    typ = pay["record_id"].map(lab).fillna("NO_LABEL")
    return (pd.crosstab(pay["reason_code"], typ).rename_axis(index="reason_code", columns="injected_error_type").reset_index())


def _first_number(text) -> float:
    try:
        return float(str(text).split("/")[0].strip())
    except ValueError:
        return np.nan


def reference_amount(kind: str, row: pd.Series, original) -> float:
    """What the injected change was worth, rebuilt from the label's original value (evaluation only)."""
    o = _first_number(original)
    dim = pd.Timestamp(row["month"]).days_in_month
    prorate = row["payable_days"] / dim
    if kind == "DOUBLE_PAYMENT":
        return float(row["net_pay"])
    if kind == "MISSED_DEDUCTION":
        return o
    if kind == "NEGATIVE_NET":
        return float(row["tds"]) - o
    if kind == "OVERTIME_SPIKE":
        return float(row["overtime_pay"]) - o
    if kind == "ROLE_MISMATCH":
        return (float(row["monthly_fixed_gross"]) - o) * prorate
    if kind == "STALE_SALARY":
        return o - float(row["gross_pay"])
    if kind == "WRONG_DECIMAL":
        return float(row["gross_pay"]) - o
    if kind == "ZERO_PAY":
        return o                                  # label stores the original GROSS, not the net, so this is an upper reference
    return np.nan


def money_vs_reference(official: pd.DataFrame, payroll_raw: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per injected error type: how the estimated amount compares with the reference amount (flagged, labelled rows only)."""
    pay = official[official["source_dataset"] == "payroll_monthly_features"].copy()
    raw = payroll_raw.drop_duplicates("payroll_item_id").set_index("payroll_item_id")
    lab = labels.set_index(LABEL_ID)
    rows = []
    for kind, grp in lab.groupby(LABEL_TYPE):
        for pid, lrow in grp.iterrows():
            hit = pay[pay["record_id"] == pid]
            if hit.empty or pid not in raw.index:
                continue
            r = raw.loc[pid]
            est = float(hit["money_at_risk"].iloc[0]) if pd.notna(hit["money_at_risk"].iloc[0]) else np.nan
            if kind == "DOUBLE_PAYMENT":              # the estimate is split over both copies: compare the pair total
                same = pay[(pay["employee_id"] == r["employee_id"]) & (pay["period"] == hit["period"].iloc[0]) & (pay["reason_code"] == "DUPLICATE_PAYMENT")]
                est = float(same["money_at_risk"].sum()) if len(same) else np.nan
            rows.append({"injected_error_type": kind, "record_id": pid, "reason_code": hit["reason_code"].iloc[0],
                         "estimate": est, "reference": reference_amount(kind, r, lrow["original_value"])})
    d = pd.DataFrame(rows, columns=["injected_error_type", "record_id", "reason_code", "estimate", "reference"])
    d["ratio_estimate_to_reference"] = d["estimate"] / d["reference"].where(d["reference"] > 0)
    summary = d.groupby("injected_error_type").agg(
        flagged_and_labelled=("record_id", "size"), with_estimate=("estimate", lambda s: int(s.notna().sum())),
        median_estimate=("estimate", "median"), median_reference=("reference", "median"),
        median_ratio=("ratio_estimate_to_reference", "median"),
        within_10pct=("ratio_estimate_to_reference", lambda s: int(((s - 1).abs() <= 0.10).sum())),
    ).reset_index()
    return summary
