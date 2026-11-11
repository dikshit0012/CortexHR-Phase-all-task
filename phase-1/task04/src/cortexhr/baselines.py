"""Helpers for Day 3 / Task 3: honest baseline models with stratified 5-fold cross-validation.

Design rules (each one protects against a specific kind of self-deception):

* The feature pipeline lives INSIDE every model (``Pipeline([features, model])``), so imputers,
  scalers and encoders are fitted on the training part of each fold only.
* One list of folds is built once and passed to every model, so all models are scored on exactly
  the same train/test splits.
* Scores come from out-of-fold predictions (``cross_val_predict``): every row is predicted by a
  model that never saw it.
* Any honest cross-validated ROC-AUC above 0.9 raises an error: go and look for leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .audit import TARGET, load_attrition
from .features import build_pipeline, prepare_training_rows, select_features
from .monthly import add_monthly_features, load_monthly_tables
from .paths import WORKBOOK

RANDOM_STATE = 42
N_SPLITS = 5
SUSPICIOUS_AUC = 0.9
THRESHOLD = 0.5


class SuspiciouslyGoodScoreError(RuntimeError):
    """An honest cross-validated score is high enough to suggest data leakage."""


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load_model_table(path=WORKBOOK, include_unconfirmed: bool = False, use_monthly: bool = True):
    """Task 1 row rules + Task 2 monthly features -> ``(X, y, source)``.

    ``X`` holds only allowed input columns (the target and leaks are refused by ``select_features``).
    ``source`` says whether a row is a master-snapshot or a simulated historical row; it is for
    reporting only and is never a model input.
    """
    raw = load_attrition(path)
    rows, _ = prepare_training_rows(raw)
    rows = rows.reset_index(drop=True)
    if use_monthly:
        attendance, payroll = load_monthly_tables(path)
        table, _ = add_monthly_features(rows, attendance, payroll)
    else:
        table = rows
    X = select_features(table, include_unconfirmed=include_unconfirmed, use_monthly=use_monthly)
    return X, table[TARGET].copy(), table["source"].copy()


# ----------------------------------------------------------------------------------------------
# folds and models
# ----------------------------------------------------------------------------------------------
def make_folds(y: pd.Series, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE):
    """Stratified folds as a list of (train_idx, test_idx). Build ONCE, reuse for every model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(np.zeros(len(y)), y))
    seen = np.concatenate([test for _, test in folds])
    if len(seen) != len(y) or len(np.unique(seen)) != len(y):
        raise AssertionError("Folds must test every row exactly once.")
    return folds


def make_models() -> dict[str, object]:
    """The baseline classifiers (unfitted). Names are used in every table."""
    return {
        "Dummy (always stays)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (balanced)": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree (no depth limit, balanced)": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree (depth 4, balanced)": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE),
    }


def model_pipeline(classifier, **pipeline_kwargs) -> Pipeline:
    """Feature pipeline + classifier, so preprocessing is re-fitted inside every fold."""
    return Pipeline([("features", build_pipeline(**pipeline_kwargs)), ("model", classifier)])


# ----------------------------------------------------------------------------------------------
# predictions and scores
# ----------------------------------------------------------------------------------------------
def out_of_fold_proba(estimator, X: pd.DataFrame, y: pd.Series, folds) -> np.ndarray:
    """Probability of leaving for every row, predicted by a model that never saw that row."""
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("y must contain exactly the classes 0 and 1 (1 = left within 12 months).")
    proba = cross_val_predict(estimator, X, y, cv=folds, method="predict_proba")
    return proba[:, 1]          # classes are sorted [0, 1], so column 1 is P(left)


def score_predictions(y_true, p, threshold: float = THRESHOLD) -> dict[str, float]:
    """Ranking scores (from probabilities) and yes/no scores (after applying ``threshold``)."""
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    yes_no = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "precision": precision_score(y_true, yes_no, zero_division=0),
        "recall": recall_score(y_true, yes_no, zero_division=0),
        "f1": f1_score(y_true, yes_no, zero_division=0),
        "accuracy": accuracy_score(y_true, yes_no),
    }


def per_fold_scores(y: pd.Series, p: np.ndarray, folds) -> pd.DataFrame:
    """Scores computed fold by fold on the out-of-fold predictions (comparable with a 'mean over 5 folds')."""
    y = np.asarray(y)
    rows = [score_predictions(y[test], p[test]) for _, test in folds]
    return pd.DataFrame(rows).rename_axis("fold")


def summarise(y: pd.Series, p: np.ndarray, folds) -> dict[str, float]:
    """One row of the comparison table: per-fold mean/sd (benchmark style) and pooled scores."""
    pf = per_fold_scores(y, p, folds)
    pooled = score_predictions(y, p)
    out = {f"{k}_mean": pf[k].mean() for k in pf.columns}
    out["roc_auc_sd"] = pf["roc_auc"].std(ddof=1)
    out["pr_auc_sd"] = pf["pr_auc"].std(ddof=1)
    out.update({f"{k}_pooled": v for k, v in pooled.items() if k in ("roc_auc", "pr_auc")})
    return out


def bootstrap_auc_interval(y, p, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for the pooled ROC-AUC (how much it moves with a different sample of rows)."""
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs = []
    while len(aucs) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def resubstitution_auc(estimator, X: pd.DataFrame, y: pd.Series) -> float:
    """ROC-AUC on the SAME rows the model was fitted on. Not an honest score: it shows memorisation."""
    est = estimator.fit(X, y)
    return float(roc_auc_score(y, est.predict_proba(X)[:, 1]))


def check_suspicious_scores(scores: dict[str, float], threshold: float = SUSPICIOUS_AUC) -> None:
    """Raise if any honest cross-validated ROC-AUC is above ``threshold`` (go and look for leakage)."""
    bad = {k: round(v, 3) for k, v in scores.items() if v > threshold}
    if bad:
        raise SuspiciouslyGoodScoreError(
            f"Cross-validated ROC-AUC above {threshold}: {bad}. Scores this high on attrition almost always mean "
            "leakage (an exit column, the target, or future data in the features). Check the feature list first.")


# ----------------------------------------------------------------------------------------------
# benchmark sheet
# ----------------------------------------------------------------------------------------------
def load_benchmarks(path=WORKBOOK) -> pd.DataFrame:
    """The reference results from the ``baseline_benchmarks`` sheet, as delivered."""
    return pd.read_excel(path, sheet_name="baseline_benchmarks")
