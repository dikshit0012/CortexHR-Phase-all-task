"""Task 3 helpers: shared folds, out-of-fold scoring, leakage checks."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RANDOM_STATE = 42
N_SPLITS = 5
LEAKAGE_ALARM = 0.90
POSITIVE_LABEL = 1


def make_folds(X, y, random_state: int = RANDOM_STATE):
    """Build the fold list ONCE; pass the same list to every model."""
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    return list(splitter.split(X, y))


def positive_column(y) -> int:
    """Index of the positive class in predict_proba output (classes are sorted)."""
    classes = list(np.unique(y))
    if POSITIVE_LABEL not in classes:
        raise ValueError(f"positive label {POSITIVE_LABEL} not in target classes {classes}")
    return classes.index(POSITIVE_LABEL)


def out_of_fold_proba(model, X, y, folds) -> np.ndarray:
    """Probability of left_within_12m == 1 for every row, from a model that never saw it."""
    proba = cross_val_predict(clone(model), X, y, cv=folds, method="predict_proba")
    return proba[:, positive_column(y)]


def out_of_fold_class(model, X, y, folds) -> np.ndarray:
    return cross_val_predict(clone(model), X, y, cv=folds, method="predict")


def safe_metrics(y_true, y_pred, y_score) -> dict:
    """Classification metrics with undefined cases returned as 0 (never an exception)."""
    return {
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


@dataclass
class Evaluation:
    name: str
    proba: np.ndarray        # out-of-fold P(left)
    pred: np.ndarray         # out-of-fold 0/1 class predictions
    pooled: dict             # metrics from all out-of-fold rows pooled together
    per_fold: pd.DataFrame   # metrics for each fold's held-out rows

    @property
    def fold_mean(self) -> pd.Series:
        return self.per_fold.mean()

    @property
    def fold_std(self) -> pd.Series:
        return self.per_fold.std(ddof=1)


def evaluate(name, model, X, y, folds) -> Evaluation:
    """Out-of-fold predictions, then metrics two ways (pooled and per fold)."""
    y_arr = np.asarray(y)
    proba = out_of_fold_proba(model, X, y, folds)
    pred = out_of_fold_class(model, X, y, folds)
    pooled = safe_metrics(y_arr, pred, proba)
    rows = [safe_metrics(y_arr[te], pred[te], proba[te]) for _, te in folds]
    return Evaluation(name, proba, pred, pooled, pd.DataFrame(rows).rename_axis("fold"))


def train_vs_validation(model, X, y, folds) -> pd.DataFrame:
    """Training ROC-AUC (on rows the model saw) vs validation ROC-AUC (rows it did not)."""
    col = positive_column(y)
    rows = []
    for i, (tr, te) in enumerate(folds):
        fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
        rows.append({
            "fold": i,
            "train_roc_auc": roc_auc_score(y.iloc[tr], fitted.predict_proba(X.iloc[tr])[:, col]),
            "validation_roc_auc": roc_auc_score(y.iloc[te], fitted.predict_proba(X.iloc[te])[:, col]),
        })
    return pd.DataFrame(rows).set_index("fold")


def leakage_alarm(roc_auc: float, label: str = "model") -> None:
    """Stop the run if an honest cross-validated score looks too good to be true."""
    if roc_auc > LEAKAGE_ALARM:
        raise RuntimeError(f"LEAKAGE ALARM: {label} ROC-AUC {roc_auc:.3f} > {LEAKAGE_ALARM}. "
                           "Investigate features and the prediction cutoff before trusting it.")
