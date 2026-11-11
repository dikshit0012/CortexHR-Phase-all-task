"""Task 7 - turn the existing Task 5 / Task 6 flags into explained records.

Nothing here fits a model or reads evaluation labels. Reasons come only from the detectors' own signals
(rule columns and the Isolation Forest flag) and from the payroll / attendance columns of the record.

Primary reason = the first fired code in reason_codes.PRIORITY (documented, deterministic).
Every other fired code is kept as a secondary reason in the detail report.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cortexhr_task5.attendance_anomaly_detector import MAX_PLAUSIBLE_AVG_HOURS, STREAK_DAYS_THRESHOLD
from cortexhr_task5.attendance_anomaly_detector import RULE_COLUMNS as ATTENDANCE_RULE_COLUMNS
from cortexhr_task6.payroll_rules import RULE_COLUMNS as PAYROLL_RULE_COLUMNS
from cortexhr_task6.payroll_rules import RuleConfig, previous_month_flag

from .money_at_risk import DUP_KEYS, candidate_amounts, duplicate_amounts
from .reason_codes import PRIORITY, PRIORITY_INDEX, REASON_CODES
from .reason_text import build_reason_text

# Columns of an explained (internal) record. An empty result (no flags) still has these columns.
INTERNAL_COLUMNS = ["_idx", "source_dataset", "record_id", "employee_id", "period", "primary_reason_code",
                    "all_reason_codes", "reason_text", "money_at_risk", "money_basis", "candidate_amounts", "rule_count",
                    "model_flag", "signal_count", "anomaly_score_raw", "anomaly_score"]
PAYROLL_SOURCE = "payroll_monthly_features"
ATTENDANCE_SOURCE = "attendance_monthly_features"
PAYROLL_CODES = [c for c in PRIORITY if REASON_CODES[c].source == "payroll"]
ATTENDANCE_CODES = [c for c in PRIORITY if REASON_CODES[c].source == "attendance"]

# Task 6 rule column -> the codes it can produce (direction / sub-condition is read from the data).
PAYROLL_RULE_MAP = {
    "rule_duplicate_payment": ["DUPLICATE_PAYMENT"],
    "rule_negative_net": ["NEGATIVE_NET_PAY"],
    "rule_zero_net": ["ZERO_PAY"],
    "rule_gross_mismatch": ["GROSS_ABOVE_NORM", "GROSS_BELOW_EXPECTED"],
    "rule_overtime_cap": ["OVERTIME_ABOVE_CAP", "OVERTIME_NOT_ELIGIBLE"],
    "rule_missed_pf": ["PF_NOT_DEDUCTED"],
    "rule_salary_drop": ["SALARY_DROP"],
    "rule_fixed_gross_jump": ["SALARY_JUMP"],
    "rule_role_pay_outlier": ["PAY_ABOVE_ROLE_NORM"],
    "rule_invalid_amounts": ["INVALID_PAYROLL_DATA"],
}
ATTENDANCE_RULE_MAP = {
    "rule_unexcused_absence": ["NO_ATTENDANCE", "LONG_UNPAID_ABSENCE"],
    "rule_implausible_hours": ["IMPLAUSIBLE_HOURS"],
}


def _check_rule_columns(flags: pd.DataFrame, rule_map: Dict[str, List[str]], upstream_rules: List[str], label: str) -> None:
    """Fail loudly if a detector rule has no explanation, or the flags carry an unmapped rule."""
    unmapped_upstream = [r for r in upstream_rules if r not in rule_map]
    present_unmapped = [c for c in flags.columns if c.startswith("rule_") and c not in rule_map and c != "rule_count"]
    missing = [r for r in rule_map if r not in flags.columns]
    if unmapped_upstream or present_unmapped or missing:
        raise ValueError(f"{label}: rule/reason mapping is out of date. Unmapped rules: "
                         f"{sorted(set(unmapped_upstream + present_unmapped))}; missing columns: {missing}")


def normalise_scores(raw_scores: pd.Series) -> pd.Series:
    """Min-max scale a detector's raw scores to 0..1 (higher = more unusual), over ALL scored rows of that dataset."""
    s = pd.to_numeric(raw_scores, errors="coerce")
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def _pick_primary(fires: pd.DataFrame, codes: List[str]) -> pd.Series:
    primary = pd.Series(pd.NA, index=fires.index, dtype="object")
    for code in codes:                                       # PRIORITY order
        take = fires[code] & primary.isna()
        primary[take] = code
    return primary


def _assert_no_unexplained(flagged: pd.Series, primary: pd.Series, ids: pd.Series, label: str) -> None:
    bad = flagged & primary.isna()
    if bad.any():
        raise ValueError(f"{label}: {int(bad.sum())} flagged record(s) have no explainable signal, e.g. {list(ids[bad][:5])}")


# ---------------------------------------------------------------------------
# Payroll (Task 6)
# ---------------------------------------------------------------------------
def explain_payroll(feats: pd.DataFrame, flags: pd.DataFrame, rule_cfg: Optional[RuleConfig] = None) -> pd.DataFrame:
    """Explain every flagged payroll record. `feats` and `flags` are the Task 6 `run_detection` outputs.

    Returns one internal row per flagged record (index = the Task 6 row index) with reason, text and money.
    """
    cfg = rule_cfg or RuleConfig()
    if not feats.index.equals(flags.index):
        raise ValueError("payroll features and flags are not aligned (different index); records cannot be matched safely")
    _check_rule_columns(flags, PAYROLL_RULE_MAP, PAYROLL_RULE_COLUMNS, "payroll")

    fires = pd.DataFrame(False, index=feats.index, columns=PAYROLL_CODES)
    r = flags
    gross, expected = feats["gross_pay"], feats["expected_gross"]
    hours = feats["overtime_hours_paid"]
    fires["DUPLICATE_PAYMENT"] = r["rule_duplicate_payment"] == 1
    fires["NEGATIVE_NET_PAY"] = r["rule_negative_net"] == 1
    fires["ZERO_PAY"] = r["rule_zero_net"] == 1
    fires["GROSS_ABOVE_NORM"] = (r["rule_gross_mismatch"] == 1) & (gross > expected)
    fires["GROSS_BELOW_EXPECTED"] = (r["rule_gross_mismatch"] == 1) & (gross < expected)
    fires["OVERTIME_ABOVE_CAP"] = (r["rule_overtime_cap"] == 1) & (hours > cfg.overtime_cap_hours)
    fires["OVERTIME_NOT_ELIGIBLE"] = (r["rule_overtime_cap"] == 1) & (hours > 0) & (feats["overtime_eligible"] != "Y")
    fires["PF_NOT_DEDUCTED"] = r["rule_missed_pf"] == 1
    fires["SALARY_DROP"] = r["rule_salary_drop"] == 1
    fires["SALARY_JUMP"] = r["rule_fixed_gross_jump"] == 1
    fires["PAY_ABOVE_ROLE_NORM"] = r["rule_role_pay_outlier"] == 1
    fires["INVALID_PAYROLL_DATA"] = r["rule_invalid_amounts"] == 1
    fires["STATISTICAL_OUTLIER"] = (r["if_flag"] == 1) & (r["rules_only_flag"] == 0)      # model-only
    # a rule that fired but whose direction could not be read from the data (e.g. gross == expected) is not silently dropped
    unresolved_gross = (r["rule_gross_mismatch"] == 1) & ~fires[["GROSS_ABOVE_NORM", "GROSS_BELOW_EXPECTED"]].any(axis=1)
    fires.loc[unresolved_gross, "GROSS_ABOVE_NORM"] = True

    # context for the explanation only (the flag and the amount are unchanged): was last month's pay itself flagged as a spike?
    prev_spike = previous_month_flag(feats, fires["PAY_ABOVE_ROLE_NORM"] | fires["SALARY_JUMP"])

    flagged = r["combined_flag"] == 1
    primary = _pick_primary(fires, PAYROLL_CODES)
    _assert_no_unexplained(flagged, primary, feats["payroll_item_id"], "payroll")

    cands = candidate_amounts(feats, flags, cfg.overtime_cap_hours)
    _, dup_total = duplicate_amounts(feats, r["rule_duplicate_payment"])
    col_idx = {c: i for i, c in enumerate(cands.columns)}
    money = pd.Series(np.nan, index=feats.index)
    arr = cands.to_numpy()
    for code in PAYROLL_CODES:
        rows = (primary == code).to_numpy()
        money[rows] = arr[rows, col_idx[code]]

    sel = flagged[flagged].index
    dup_copies = feats[DUP_KEYS].notna().all(axis=1)
    copies_all = pd.Series(1, index=feats.index)
    if dup_copies.any():
        copies_all.loc[dup_copies] = feats[dup_copies].groupby(DUP_KEYS, sort=False)["employee_id"].transform("size")

    rows_out = []
    for idx in sel:
        rec = feats.loc[idx].to_dict()
        rec.update({"period": feats.at[idx, "month"], "money": money.at[idx], "dup_total": dup_total.at[idx],
                    "copies": copies_all.at[idx], "overtime_cap": cfg.overtime_cap_hours,
                    "prev_month_spike": bool(prev_spike.at[idx])})
        code = primary.at[idx]
        fired = [c for c in PAYROLL_CODES if fires.at[idx, c]]
        rows_out.append({
            "_idx": idx, "source_dataset": PAYROLL_SOURCE, "record_id": feats.at[idx, "payroll_item_id"],
            "employee_id": feats.at[idx, "employee_id"], "period": feats.at[idx, "month"],
            "primary_reason_code": code, "all_reason_codes": ";".join(fired),
            "reason_text": build_reason_text(code, rec), "money_at_risk": money.at[idx],
            "money_basis": REASON_CODES[code].money_basis,
            "candidate_amounts": "; ".join(f"{c}={cands.at[idx, c]:.2f}" for c in fired if pd.notna(cands.at[idx, c])),
            "rule_count": int(r.at[idx, "rule_count"]), "model_flag": int(r.at[idx, "if_flag"]),
            "signal_count": int(r.at[idx, "rule_count"]) + int(r.at[idx, "if_flag"]),
            "anomaly_score_raw": float(r.at[idx, "if_anomaly_score"]),
        })
    out = pd.DataFrame(rows_out, columns=INTERNAL_COLUMNS).set_index("_idx")
    out.index.name = None
    norm = normalise_scores(flags["if_anomaly_score"])
    out["anomaly_score"] = norm.reindex(out.index).round(3)
    return out


# ---------------------------------------------------------------------------
# Attendance (Task 5)
# ---------------------------------------------------------------------------
def explain_attendance(feats: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Explain every flagged attendance record. `feats`/`flags` come from Task 5's `run_detector`. Money is never estimated."""
    if not feats.index.equals(flags.index):
        raise ValueError("attendance features and flags are not aligned (different index); records cannot be matched safely")
    _check_rule_columns(flags, ATTENDANCE_RULE_MAP, ATTENDANCE_RULE_COLUMNS, "attendance")

    sched = feats["scheduled_days"].fillna(0)
    present = feats["present_days"].fillna(0)
    paid = feats["paid_leave_days"].fillna(0)
    streak = feats["longest_unpaid_streak_days"].fillna(0)
    hours = feats["avg_hours_worked"].fillna(0)

    fires = pd.DataFrame(False, index=feats.index, columns=ATTENDANCE_CODES)
    fires["NO_ATTENDANCE"] = (sched > 0) & (present == 0) & (paid < sched)
    fires["LONG_UNPAID_ABSENCE"] = streak >= STREAK_DAYS_THRESHOLD
    fires["IMPLAUSIBLE_HOURS"] = (present > 0) & (hours > MAX_PLAUSIBLE_AVG_HOURS)
    fires["PATTERN_SHIFT"] = (flags["if_flag"] == 1) & (flags["rule_count"] == 0)
    # the split above must reproduce the Task 5 rules exactly, otherwise the explanation would be wrong
    if not ((fires["NO_ATTENDANCE"] | fires["LONG_UNPAID_ABSENCE"]).astype(int) == flags["rule_unexcused_absence"]).all() or \
       not (fires["IMPLAUSIBLE_HOURS"].astype(int) == flags["rule_implausible_hours"]).all():
        raise ValueError("attendance reason split no longer matches the Task 5 rules; update ATTENDANCE_RULE_MAP")

    flagged = flags["combined_anomaly"] == 1
    primary = _pick_primary(fires, ATTENDANCE_CODES)
    _assert_no_unexplained(flagged, primary, feats["attendance_feature_id"], "attendance")

    norm = normalise_scores(flags["if_anomaly_score"])
    rows_out = []
    for idx in flagged[flagged].index:
        rec = feats.loc[idx].to_dict()
        code = primary.at[idx]
        fired = [c for c in ATTENDANCE_CODES if fires.at[idx, c]]
        rows_out.append({
            "_idx": idx, "source_dataset": ATTENDANCE_SOURCE, "record_id": feats.at[idx, "attendance_feature_id"],
            "employee_id": feats.at[idx, "employee_id"], "period": feats.at[idx, "month"],
            "primary_reason_code": code, "all_reason_codes": ";".join(fired),
            "reason_text": build_reason_text(code, rec), "money_at_risk": np.nan,
            "money_basis": REASON_CODES[code].money_basis, "candidate_amounts": "",
            "rule_count": int(flags.at[idx, "rule_count"]), "model_flag": int(flags.at[idx, "if_flag"]),
            "signal_count": int(flags.at[idx, "rule_count"]) + int(flags.at[idx, "if_flag"]),
            "anomaly_score_raw": float(flags.at[idx, "if_anomaly_score"]),
            "anomaly_score": round(float(norm.at[idx]), 3),
        })
    out = pd.DataFrame(rows_out, columns=INTERNAL_COLUMNS).set_index("_idx")
    out.index.name = None
    return out
