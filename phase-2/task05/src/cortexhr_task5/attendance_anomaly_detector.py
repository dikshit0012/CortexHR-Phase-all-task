"""Task 5 - unsupervised attendance anomaly detector (Isolation Forest + simple rules).

Nothing in this module reads or accepts ground-truth labels. Labels are only
touched in `evaluation.py`, after predictions have been produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

ID_COL = "attendance_feature_id"
EMPLOYEE_COL = "employee_id"
MONTH_COL = "month"

# Columns the raw sheet must contain.
REQUIRED_COLUMNS = [
    ID_COL, EMPLOYEE_COL, MONTH_COL, "scheduled_days", "present_days",
    "half_days_unpaid", "paid_leave_days", "unpaid_absence_days",
    "longest_unpaid_streak_days", "late_arrivals", "avg_hours_worked",
    "overtime_hours", "attendance_rate",
]

# Features fed to Isolation Forest. Chosen up front from the data dictionary
# (no label information). Left out on purpose:
#   scheduled_days, present_days -> largely captured by attendance_rate
#   wfh_days                     -> varies by department policy, not by behaviour
#   job_level, department_id     -> identifiers / categories, not attendance
MODEL_FEATURES: List[str] = [
    "attendance_rate",
    "unpaid_absence_days",
    "longest_unpaid_streak_days",
    "half_days_unpaid",
    "paid_leave_days",
    "late_arrivals",
    "avg_hours_worked",
    "overtime_hours",
    "rate_change_vs_own_history",   # engineered from PAST months only
]

# Rule thresholds come from working-calendar / labour-hour logic, not from tuning.
STREAK_DAYS_THRESHOLD = 5       # one full working week of consecutive unpaid absence
MAX_PLAUSIBLE_AVG_HOURS = 12.0  # a monthly average above this is not plausible


@dataclass
class DetectorConfig:
    contamination: float = 0.03
    n_estimators: int = 200
    random_state: int = 42
    features: List[str] = field(default_factory=lambda: list(MODEL_FEATURES))


# --------------------------------------------------------------------------
# Validation and feature building
# --------------------------------------------------------------------------
def validate_attendance_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns and basic integrity; return a cleaned copy."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"attendance data is missing required columns: {missing}")
    if df.empty:
        raise ValueError("attendance data is empty")
    out = df.copy()
    if out[ID_COL].duplicated().any():
        raise ValueError(f"duplicate values in {ID_COL}")
    out[MONTH_COL] = pd.to_datetime(out[MONTH_COL], errors="coerce")
    if out[MONTH_COL].isna().any():
        raise ValueError("some month values could not be parsed as dates")
    numeric = [c for c in REQUIRED_COLUMNS if c not in (ID_COL, EMPLOYEE_COL, MONTH_COL)]
    for c in numeric:
        # invalid strings -> NaN, infinities -> NaN; the imputer handles them later
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `rate_change_vs_own_history`: this month's rate minus the mean rate of
    the employee's EARLIER months (current month excluded, so no look-ahead).
    An employee's first month has no history and gets 0 (neutral)."""
    out = df.sort_values([EMPLOYEE_COL, MONTH_COL]).copy()
    prior_mean = out.groupby(EMPLOYEE_COL)["attendance_rate"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    out["rate_change_vs_own_history"] = (out["attendance_rate"] - prior_mean).fillna(0.0)
    return out.sort_index()


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw monthly data and return it with engineered features."""
    return add_history_features(validate_attendance_frame(df))


# --------------------------------------------------------------------------
# Isolation Forest wrapper (labels are not an argument anywhere)
# --------------------------------------------------------------------------
class AttendanceAnomalyDetector:
    """Median-impute -> Isolation Forest. Preprocessing is fit on training rows only."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.imputer_: Optional[SimpleImputer] = None
        self.model_: Optional[IsolationForest] = None
        self.kept_features_: List[str] = []
        self.dropped_constant_: List[str] = []

    def fit(self, features: pd.DataFrame) -> "AttendanceAnomalyDetector":
        cols = self.config.features
        missing = [c for c in cols if c not in features.columns]
        if missing:
            raise ValueError(f"feature frame is missing model features: {missing}")
        X = features[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        # drop columns that are constant or entirely missing in the training data
        nunique = X.nunique(dropna=True)
        self.dropped_constant_ = [c for c in cols if nunique[c] <= 1]
        self.kept_features_ = [c for c in cols if c not in self.dropped_constant_]
        if not self.kept_features_:
            raise ValueError("no usable (non-constant) features to fit on")
        X = X[self.kept_features_]
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        self.model_ = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        ).fit(self.imputer_.transform(X))
        return self

    def _prepare(self, features: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit() before predict()")
        X = features[self.kept_features_].apply(pd.to_numeric, errors="coerce")
        return self.imputer_.transform(X.replace([np.inf, -np.inf], np.nan))

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """if_flag (1 anomaly / 0 normal) and if_anomaly_score (higher = more abnormal)."""
        Xi = self._prepare(features)
        raw = self.model_.predict(Xi)                 # sklearn: -1 anomaly, 1 normal
        score = -self.model_.decision_function(Xi)    # decision_function: lower = more abnormal
        return pd.DataFrame(
            {"if_flag": (raw == -1).astype(int), "if_anomaly_score": score},
            index=features.index,
        )

    def fit_predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).predict(features)


# --------------------------------------------------------------------------
# Business rules
# --------------------------------------------------------------------------
RULE_COLUMNS = ["rule_unexcused_absence", "rule_implausible_hours"]
RULE_REASONS = {
    "rule_unexcused_absence": "unexcused absence (long unpaid streak, or no presence without leave cover)",
    "rule_implausible_hours": "implausible average hours per day",
}


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Two interpretable rules, one 0/1 column each.

    rule_unexcused_absence
        Fires on >= 5 consecutive unpaid absence days, OR no presence at all in a
        month that had scheduled days and where paid leave does not cover the
        schedule. Paid leave is the legitimate explanation, so a month fully on paid
        leave is NOT flagged, and a month with no scheduled days is not flagged.
        Likely false positives: approved unpaid leave that was never recorded, or
        leave entered late in the HR system.

    rule_implausible_hours
        Fires when present_days > 0 and the average hours per recorded day is > 12.
        Usually a clock/entry error rather than behaviour.
        Likely false positives: on-call or shift roles with a few very long days.
    """
    sched = df["scheduled_days"].fillna(0)
    present = df["present_days"].fillna(0)
    paid_leave = df["paid_leave_days"].fillna(0)
    streak = df["longest_unpaid_streak_days"].fillna(0)
    hours = df["avg_hours_worked"].fillna(0)

    no_presence_unexcused = (sched > 0) & (present == 0) & (paid_leave < sched)
    r1 = (streak >= STREAK_DAYS_THRESHOLD) | no_presence_unexcused
    r2 = (present > 0) & (hours > MAX_PLAUSIBLE_AVG_HOURS)
    return pd.DataFrame(
        {"rule_unexcused_absence": r1.astype(int), "rule_implausible_hours": r2.astype(int)},
        index=df.index,
    )


def combine_flags(model_out: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """combined_anomaly = IF flag OR any rule. Every source column is kept."""
    out = pd.concat([model_out, rules], axis=1)
    out["rule_count"] = out[RULE_COLUMNS].sum(axis=1)
    out["rules_only_flag"] = (out["rule_count"] > 0).astype(int)
    out["combined_anomaly"] = ((out["if_flag"] == 1) | (out["rule_count"] > 0)).astype(int)

    def reasons(row) -> str:
        parts = []
        if row["if_flag"] == 1:
            parts.append("isolation forest: unusual combination of attendance features")
        for c in RULE_COLUMNS:
            if row[c] == 1:
                parts.append(RULE_REASONS[c])
        return "; ".join(parts)

    out["anomaly_reasons"] = out.apply(reasons, axis=1)
    return out


# --------------------------------------------------------------------------
# HR ranking (no labels)
# --------------------------------------------------------------------------
def rank_for_review(flags: pd.DataFrame, flagged_months_per_employee: Optional[pd.Series] = None) -> pd.DataFrame:
    """Points-based priority for FLAGGED rows only.

    points = 1 if Isolation Forest flagged
           + 1 more if its score is in the top third of IF-flagged scores
           + 2 per business rule triggered
           + 1 if the employee has >= 2 flagged months (repeat pattern)
    High >= 4, Medium 2-3, Low 1. Within equal points, sorted by IF score (desc).
    Points are ordinal, not probabilities; flags are investigation leads only.
    """
    f = flags[flags["combined_anomaly"] == 1].copy()
    if f.empty:
        f["priority_points"] = pd.Series(dtype=int)
        f["priority"] = pd.Series(dtype=object)
        return f
    if_scores = f.loc[f["if_flag"] == 1, "if_anomaly_score"]
    top_third_cut = if_scores.quantile(2 / 3) if len(if_scores) else np.inf
    pts = f["if_flag"] + ((f["if_flag"] == 1) & (f["if_anomaly_score"] >= top_third_cut)).astype(int)
    pts = pts + 2 * f["rule_count"]
    if flagged_months_per_employee is not None:
        pts = pts + (flagged_months_per_employee.reindex(f.index).fillna(0) >= 2).astype(int)
    f["priority_points"] = pts.astype(int)
    f["priority"] = pd.cut(f["priority_points"], bins=[-1, 1, 3, 100], labels=["Low", "Medium", "High"])
    return f.sort_values(["priority_points", "if_anomaly_score"], ascending=False)


def employee_flag_counts(df: pd.DataFrame, flags: pd.DataFrame) -> pd.Series:
    """Number of flagged months of the same employee, aligned to df's index."""
    tmp = pd.DataFrame({EMPLOYEE_COL: df[EMPLOYEE_COL], "f": flags["combined_anomaly"]})
    return tmp.groupby(EMPLOYEE_COL)["f"].transform("sum")


def run_detector(raw: pd.DataFrame, config: Optional[DetectorConfig] = None):
    """End to end and label-free. Returns (feature_frame, flags, ranked, detector)."""
    feats = build_feature_frame(raw)
    det = AttendanceAnomalyDetector(config)
    model_out = det.fit_predict(feats)
    flags = combine_flags(model_out, apply_rules(feats))
    ranked = rank_for_review(flags, employee_flag_counts(feats, flags))
    return feats, flags, ranked, det
