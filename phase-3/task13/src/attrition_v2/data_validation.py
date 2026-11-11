"""Phase 2 + 4: data/leakage audit and verification that the Task 3 folds are reused unchanged."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config
from cortexhr.audit import LEAKAGE_COLUMNS, load_attrition
from cortexhr.baselines import N_SPLITS as T3_N_SPLITS, RANDOM_STATE as T3_SEED, load_model_table, make_folds
from cortexhr.features import (FORBIDDEN_REASONS, UNCONFIRMED_FEATURES, check_no_forbidden, feature_columns,
                               prepare_training_rows)


class TargetError(ValueError):
    """The target column is missing or not a clean 0/1 label."""


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load_data(path=None):
    """Task 1 row rules + Task 2 monthly features -> ``(X, y, source)`` exactly as Task 3 built them."""
    X, y, source = load_model_table(path or config.WORKBOOK)
    validate_target(y)
    return X, y, source


def validate_target(y: pd.Series, name: str = config.TARGET) -> None:
    if y is None or len(y) == 0:
        raise TargetError("The target is empty.")
    if y.isna().any():
        raise TargetError(f"Target '{name}' has missing values.")
    if set(np.unique(y)) != {0, 1}:
        raise TargetError(f"Target '{name}' must contain exactly the classes 0 and 1, got {sorted(set(y))}.")


def target_column_in_workbook(path=None) -> str:
    """Verify (do not assume) that the label column is still called ``left_within_12m``."""
    raw = load_attrition(path or config.WORKBOOK)
    if config.TARGET not in raw.columns:
        cands = [c for c in raw.columns if "left" in c.lower()]
        raise TargetError(f"'{config.TARGET}' not found. Similar columns: {cands}")
    return config.TARGET


# ----------------------------------------------------------------------------------------------
# Phase 2: audit
# ----------------------------------------------------------------------------------------------
def dataset_audit(path=None) -> dict:
    """Dimensions, target distribution, missing values, duplicates, types, imbalance (raw sheet + model table)."""
    path = path or config.WORKBOOK
    raw = load_attrition(path)
    content = [c for c in raw.columns if c != "row_id"]
    kept, prep = prepare_training_rows(raw)
    X, y, source = load_data(path)
    miss = X.isna().mean().sort_values(ascending=False)
    return {
        "raw_shape": list(raw.shape),
        "model_table_shape": list(X.shape),
        "target": config.TARGET,
        "target_counts_raw": {int(k): int(v) for k, v in raw[config.TARGET].value_counts().items()},
        "target_rate_raw": float(raw[config.TARGET].mean()),
        "target_counts_model": {int(k): int(v) for k, v in y.value_counts().items()},
        "target_rate_model": float(y.mean()),
        "imbalance_ratio_stay_to_leave": float((1 - y.mean()) / y.mean()),
        "duplicate_rows_raw": int(raw.duplicated(subset=content).sum()),
        "duplicate_employee_ids_in_model_rows": int(kept["employee_id"].duplicated().sum()),
        "row_rules": {k: v for k, v in prep.items() if k != "negative_tenure_ids"},
        "negative_tenure_ids": prep["negative_tenure_ids"],
        "missing_share_model_inputs_top": {k: round(float(v), 4) for k, v in miss[miss > 0].head(12).items()},
        "n_input_columns_with_missing": int((miss > 0).sum()),
        "dtypes_model_inputs": {k: int(v) for k, v in X.dtypes.astype(str).value_counts().items()},
        "source_counts": {k: int(v) for k, v in source.value_counts().items()},
        "target_rate_by_source": {k: round(float(v), 4) for k, v in y.groupby(source).mean().items()},
    }


# ----------------------------------------------------------------------------------------------
# Phase 2: leakage
# ----------------------------------------------------------------------------------------------
def excluded_columns_table() -> pd.DataFrame:
    """Every column the Task 2 pipeline refuses, with the reason (documented in the README)."""
    rows = [{"column": c,
             "category": ("target" if c == config.TARGET else
                          "leakage (exists only after exit)" if c in LEAKAGE_COLUMNS else "identifier / non-feature"),
             "reason": FORBIDDEN_REASONS[c]} for c in FORBIDDEN_REASONS]
    rows += [{"column": c, "category": "unconfirmed (open item O2)",
              "reason": "not confirmed to use only exits before the observation date; off by default"}
             for c in UNCONFIRMED_FEATURES]
    return pd.DataFrame(rows)


def assert_no_leakage_columns(X: pd.DataFrame) -> None:
    """Raise if any forbidden or unconfirmed column is among the model inputs, or the list drifted."""
    check_no_forbidden(list(X.columns), "Task 13 model inputs")
    unconfirmed = [c for c in UNCONFIRMED_FEATURES if c in X.columns]
    if unconfirmed:
        raise ValueError(f"Unconfirmed feature(s) in the Task 13 inputs: {unconfirmed}")
    expected = sum(feature_columns().values(), [])
    if sorted(expected) != sorted(X.columns):
        raise ValueError("Model inputs differ from the Task 2 safe-feature list.")


def single_feature_auc_scan(X: pd.DataFrame, y: pd.Series, folds) -> pd.DataFrame:
    """Out-of-fold ROC-AUC of a shallow tree on ONE input column at a time.

    A column near 1.0 would be an almost certain leak. Categorical columns are label-coded (crude but adequate).
    """
    from sklearn.tree import DecisionTreeClassifier
    rows = []
    for c in X.columns:
        s = X[c]
        if s.dtype == object or str(s.dtype) in ("string", "category"):
            s = pd.Series(pd.factorize(s.fillna("Unknown"))[0], index=X.index)
        s = pd.to_numeric(s, errors="coerce")
        col = s.fillna(s.median()).to_frame(c)
        p = cross_val_predict(DecisionTreeClassifier(max_depth=2, min_samples_leaf=30, random_state=config.RANDOM_STATE),
                              col, y, cv=folds, method="predict_proba")[:, 1]
        rows.append({"feature": c, "single_feature_cv_auc": float(roc_auc_score(y, p))})
    out = pd.DataFrame(rows).sort_values("single_feature_cv_auc", ascending=False).reset_index(drop=True)
    out["suspicious_gt_0.9"] = out["single_feature_cv_auc"] > 0.9
    return out


def label_shuffle_control(make_estimator, X, y, folds, seed: int = 0) -> float:
    """Pooled out-of-fold ROC-AUC after shuffling the labels. Should be about 0.5; much higher means leakage."""
    rng = np.random.default_rng(seed)
    y_shuf = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    p = cross_val_predict(make_estimator(), X, y_shuf, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(y_shuf, p))


# ----------------------------------------------------------------------------------------------
# Phase 4: folds
# ----------------------------------------------------------------------------------------------
def fold_fingerprint(folds) -> str:
    h = hashlib.sha256()
    for tr, te in folds:
        h.update(np.asarray(tr, dtype=np.int64).tobytes())
        h.update(np.asarray(te, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def get_task3_folds(y: pd.Series):
    """The Task 3 folds, rebuilt with Task 3's own ``make_folds`` (Task 3 did not save the indices)."""
    if (T3_N_SPLITS, T3_SEED) != (config.N_SPLITS, config.RANDOM_STATE):
        raise AssertionError("Task 3 fold configuration changed; Task 13 must not silently follow it.")
    return make_folds(y)


def verify_folds(folds, y: pd.Series) -> dict:
    """Number of folds, coverage, stratification, and equality with an independent StratifiedKFold rebuild."""
    n = len(y)
    test_all = np.concatenate([te for _, te in folds])
    independent = list(StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True,
                                       random_state=config.RANDOM_STATE).split(np.zeros(n), y))
    same = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(folds, independent))
    overlap = any(len(np.intersect1d(tr, te)) for tr, te in folds)
    return {
        "n_folds": len(folds), "seed": config.RANDOM_STATE, "stratified": True, "shuffle": True,
        "every_row_tested_once": bool(len(test_all) == n and len(np.unique(test_all)) == n),
        "train_test_overlap": bool(overlap),
        "identical_to_independent_StratifiedKFold_rebuild": bool(same),
        "fold_sizes_test": [int(len(te)) for _, te in folds],
        "fold_positive_rates": [round(float(y.iloc[te].mean()), 4) for _, te in folds],
        "fold_fingerprint": fold_fingerprint(folds),
    }
