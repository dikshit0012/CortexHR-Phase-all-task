"""Task 6 - evaluation against the hidden payroll error labels.

This module is the ONLY place that touches ground-truth labels, and it is only
called after predictions already exist.

Label source: the `unlock_CX-06` vault sheet if the workbook has one, otherwise the
`payroll_anomaly_labels` sheet (same fallback Task 5 used for `unlock_CX-05`).

Label structure (verified by `validate_label_join`): one row per labelled payslip,
one of eight error types per payslip -> the eight types are MUTUALLY EXCLUSIVE and a
payslip has at most one label. Each type is therefore evaluated one-vs-clean.

Metric conventions
  * Precision, recall and F1 are returned as NaN (not 0) when undefined
    (no predictions / no actual cases), so absence is never reported as a score.
  * F1 = 0.0 when precision and recall are both defined and both 0.
  * Per-type `precision` (primary): TP / (TP + FP_clean). FP_clean = flagged payslips with
    NO error label. Flags that land on a payslip carrying a DIFFERENT error type are
    neither TP nor FP for this type (they are real errors, just of another kind) and are
    counted in `flagged_on_other_error_types`.
  * 95% Wilson intervals are given for recall and precision (`*_ci95_low/high`); they are wide for small types.
  * `precision_strict` counts every flag outside the type as a false positive:
    TP / all flagged payslips. Use it for a worst-case view.
  * The "ALL_ERRORS" row pools the eight types. Its precision TP / flagged is the same
    definition as the `anomaly_baselines` sheet, so it is the only directly comparable row.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from .payroll_features import ID_COL, EMPLOYEE_COL, MONTH_COL, safe_divide

LABEL_TYPE_COL = "anomaly_type"
LABEL_SHEETS = ["unlock_CX-06", "payroll_anomaly_labels"]
ALL_ERRORS = "ALL_ERRORS"


# --------------------------------------------------------------------------
# Labels and join validation
# --------------------------------------------------------------------------
def load_payroll_labels(xlsx_path: Union[str, Path], sheet: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Load labels; returns (labels, sheet_name_used). Tries unlock_CX-06 first."""
    candidates = [sheet] if sheet else LABEL_SHEETS
    # 'with' closes the file handle; an open handle blocks deleting/editing the workbook on Windows
    with pd.ExcelFile(xlsx_path) as xls:
        available = list(xls.sheet_names)
        for s in candidates:
            if s in available:
                df = pd.read_excel(xls, sheet_name=s)
                if ID_COL in df.columns and LABEL_TYPE_COL in df.columns:
                    return df, s
    xls_names = available
    raise ValueError(
        f"no labels sheet with columns '{ID_COL}' and '{LABEL_TYPE_COL}'. Tried {candidates}; "
        f"available: {xls_names}"
    )


def validate_label_join(feature_ids: pd.Series, labels: pd.DataFrame) -> Dict[str, int]:
    """Check the label -> feature join on payroll_item_id and raise on any problem.

    Reports: label rows, unique label keys, duplicate label keys, duplicate feature keys,
    matched, unmatched, and rows used for evaluation.
    """
    dup_label_keys = int(labels[ID_COL].duplicated().sum())
    dup_feature_keys = int(feature_ids.duplicated().sum())
    matched = int(labels[ID_COL].isin(set(feature_ids)).sum())
    report = {
        "label_rows": int(len(labels)),
        "unique_label_keys": int(labels[ID_COL].nunique()),
        "duplicate_label_keys": dup_label_keys,
        "duplicate_feature_keys": dup_feature_keys,
        "matched_rows": matched,
        "unmatched_rows": int(len(labels) - matched),
        "feature_rows": int(len(feature_ids)),
        "evaluation_rows_used": matched,
    }
    if dup_label_keys or dup_feature_keys:
        raise ValueError(f"duplicate join keys would duplicate evaluation records: {report}")
    if report["unmatched_rows"]:
        raise ValueError(f"labelled rows missing from the feature data: {report}")
    return report


# --------------------------------------------------------------------------
# Metric primitives
# --------------------------------------------------------------------------
def metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Precision / recall / F1 with NaN for undefined values (see module docstring)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isnan(precision) or np.isnan(recall):
        f1 = np.nan
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion; (NaN, NaN) when there are no trials.

    With only 5-15 cases per error type, a recall of 1.0 (e.g. 5/5) still has a wide interval.
    """
    if trials <= 0:
        return (np.nan, np.nan)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _as_id_set(flag: pd.Series) -> Set[str]:
    """Flag series indexed by payroll_item_id -> set of flagged ids."""
    return set(flag.index[flag.astype(int) == 1])


def per_type_metrics(approach: str, flag: pd.Series, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per error type (plus ALL_ERRORS) for one detection approach.

    `flag` is a 0/1 Series indexed by payroll_item_id covering ALL payslips.
    """
    all_ids = set(flag.index)
    flagged = _as_id_set(flag)
    labelled = set(labels[ID_COL])
    clean = all_ids - labelled
    fp_clean = len(flagged & clean)
    tn_clean = len(clean - flagged)
    rows = []
    types = sorted(labels[LABEL_TYPE_COL].unique())
    for t in types + [ALL_ERRORS]:
        pos = labelled if t == ALL_ERRORS else set(labels.loc[labels[LABEL_TYPE_COL] == t, ID_COL])
        tp = len(flagged & pos)
        fn = len(pos - flagged)
        other = len((flagged & labelled) - pos)
        m = metrics_from_counts(tp, fp_clean, fn)
        r_lo, r_hi = wilson_interval(tp, tp + fn)
        p_lo, p_hi = wilson_interval(tp, tp + fp_clean)
        rows.append({
            "approach": approach,
            "error_type": t,
            "actual_cases": len(pos),
            "predicted_positive": tp + fp_clean,
            "tp": tp, "fp": fp_clean, "fn": fn,
            "tn": tn_clean,
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "recall_ci95_low": r_lo, "recall_ci95_high": r_hi,
            "precision_ci95_low": p_lo, "precision_ci95_high": p_hi,
            "precision_strict": tp / len(flagged) if flagged else np.nan,
            "flagged_on_other_error_types": other,
            "support": len(pos),
        })
    return pd.DataFrame(rows)


def results_table(per_type: pd.DataFrame, approach_map: Dict[str, str]) -> pd.DataFrame:
    """Wide table: error type x (Model / Rules / Combined) precision and recall.

    approach_map maps display prefix -> approach name used in `per_type`.
    """
    base = per_type[per_type["approach"] == list(approach_map.values())[0]][["error_type", "actual_cases"]]
    out = base.set_index("error_type")
    for prefix, name in approach_map.items():
        sub = per_type[per_type["approach"] == name].set_index("error_type")
        out[f"{prefix} precision"] = sub["precision"]
        out[f"{prefix} recall"] = sub["recall"]
    return out.reset_index()


# --------------------------------------------------------------------------
# Rules vs model analysis
# --------------------------------------------------------------------------
def overlap_table(rules_flag: pd.Series, model_flag: pd.Series, labels: pd.DataFrame) -> pd.DataFrame:
    """For every error type: caught by rules only / model only / both / missed by both."""
    r, m = _as_id_set(rules_flag), _as_id_set(model_flag)
    rows = []
    for t in sorted(labels[LABEL_TYPE_COL].unique()) + [ALL_ERRORS]:
        pos = set(labels[ID_COL]) if t == ALL_ERRORS else set(labels.loc[labels[LABEL_TYPE_COL] == t, ID_COL])
        rows.append({
            "error_type": t,
            "actual_cases": len(pos),
            "caught_by_rules_only": len((pos & r) - m),
            "caught_by_model_only": len((pos & m) - r),
            "caught_by_both": len(pos & r & m),
            "missed_by_both": len(pos - r - m),
        })
    return pd.DataFrame(rows)


def clean_overlap(rules_flag: pd.Series, model_flag: pd.Series, labels: pd.DataFrame) -> Dict[str, int]:
    """Same four buckets for payslips with NO error label (these are false alarms)."""
    r, m = _as_id_set(rules_flag), _as_id_set(model_flag)
    clean = set(rules_flag.index) - set(labels[ID_COL])
    return {
        "clean_rows": len(clean),
        "flagged_by_rules_only": len((clean & r) - m),
        "flagged_by_model_only": len((clean & m) - r),
        "flagged_by_both": len(clean & r & m),
        "flagged_by_neither": len(clean - r - m),
    }


def rule_hit_table(flags: pd.DataFrame, labels: pd.DataFrame, rule_columns: Iterable[str]) -> pd.DataFrame:
    """For each rule: total flags, flags on clean rows, and hits per error type (index = payroll_item_id)."""
    lab = labels.set_index(ID_COL)[LABEL_TYPE_COL]
    rows = []
    for rule in rule_columns:
        fired = flags.index[flags[rule] == 1]
        types = lab.reindex(fired)
        row = {"rule": rule, "total_flags": len(fired), "flags_on_clean_rows": int(types.isna().sum())}
        for t in sorted(lab.unique()):
            row[f"hits_{t}"] = int((types == t).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Baseline replica (to verify metric definitions against `anomaly_baselines`)
# --------------------------------------------------------------------------
def baseline_replica_flag(raw: pd.DataFrame, contamination: float = 0.025, seed: int = 42,
                          dup_keep: Union[str, bool] = "first") -> pd.Series:
    """Re-run the published baseline recipe: Isolation Forest on
    gross_vs_fixed, ded_ratio, net_ratio, overtime_hours_paid, dup_flag, net_pay (median-imputed).

    The sheet does not define dup_flag; `dup_keep` picks the interpretation
    ('first' = mark 2nd+ copies of employee+month, False = mark every copy).
    Returns a 0/1 Series indexed by payroll_item_id.
    """
    X = pd.DataFrame({
        "gross_vs_fixed": safe_divide(raw["gross_pay"], raw["monthly_fixed_gross"]),
        "ded_ratio": safe_divide(raw["total_deductions"], raw["gross_pay"]),
        "net_ratio": safe_divide(raw["net_pay"], raw["gross_pay"]),
        "overtime_hours_paid": pd.to_numeric(raw["overtime_hours_paid"], errors="coerce"),
        "dup_flag": raw.duplicated([EMPLOYEE_COL, MONTH_COL], keep=dup_keep).astype(int),
        "net_pay": pd.to_numeric(raw["net_pay"], errors="coerce"),
    })
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    pred = IsolationForest(contamination=contamination, random_state=seed).fit_predict(Xi)
    return pd.Series((pred == -1).astype(int), index=raw[ID_COL].values)
