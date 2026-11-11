"""Evaluation helpers for Day 4 / Task 4: honest metrics, calibration, lift and risk bands.

NOTE: the task refers to a provided file ``cortexhr_helpers.py`` with ``binary_report`` and ``risk_bands``.
That file was not available when this module was written, so these are independent implementations
using the same names. If your team's helper file differs, compare signatures before swapping.

Terms used throughout
* score        - the model's number for each person (higher = more likely to leave).
* top k%       - the k% of people with the highest scores; ties are broken by input order so the
                 group size is exact.
* lift         - how many times more leavers a group contains than a random group of the same size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

BAND_LABELS = ("Low", "Medium", "High")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _check_inputs(y_true, score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError(f"y_true and score must be 1-D and the same length, got {y.shape} and {s.shape}.")
    if not set(np.unique(y)) <= {0, 1} or y.sum() == 0 or y.sum() == len(y):
        raise ValueError("y_true must contain both classes, coded 0 and 1.")
    if np.isnan(s).any():
        raise ValueError("score contains NaN.")
    return y, s


def top_k_mask(score, frac: float, tie_seed: int | None = None) -> np.ndarray:
    """Boolean mask of the ``frac`` highest scores (exactly round(frac*n) people).

    Ties are broken by input order, or randomly if ``tie_seed`` is given. Use ``tie_seed`` when many scores
    are identical (e.g. a Dummy model): input order can be systematically unrepresentative, a random pick is not.
    """
    s = np.asarray(score, dtype=float)
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1].")
    k = max(1, int(round(frac * len(s))))
    if tie_seed is None:
        order = np.argsort(-s, kind="stable")
    else:
        perm = np.random.default_rng(tie_seed).permutation(len(s))
        order = perm[np.argsort(-s[perm], kind="stable")]
    mask = np.zeros(len(s), dtype=bool)
    mask[order[:k]] = True
    return mask


# ----------------------------------------------------------------------------------------------
# the metrics report
# ----------------------------------------------------------------------------------------------
def binary_report(y_true, score, top_frac: float = 0.10, threshold: float = 0.5, tie_seed: int | None = None) -> dict[str, float]:
    """Metrics for a binary risk score. Every key is explained in the Day 4 notebook.

    Ranking metrics (use the whole list of scores, no cut-off needed):
      roc_auc, pr_auc, pr_auc_baseline (= share of leavers, what a useless model gets)
    Top-k% metrics (the list HR would actually review):
      top_k_people, precision_at_top, recall_at_top, lift_at_top
    Yes/no metrics at ``threshold`` (how the score behaves as an alarm):
      precision, recall, f1, accuracy, flagged_share
    Probability quality (only meaningful if ``score`` is meant to be a probability):
      brier, brier_baseline, log_loss (NaN when score is outside 0-1)
    """
    y, s = _check_inputs(y_true, score)
    prevalence = float(y.mean())
    top = top_k_mask(s, top_frac, tie_seed)
    yes = s >= threshold
    in_range = bool(s.min() >= 0 and s.max() <= 1)
    return {
        "n": int(len(y)), "leavers": int(y.sum()), "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)), "pr_auc_baseline": prevalence,
        "top_frac": float(top_frac), "top_k_people": int(top.sum()),
        "precision_at_top": float(y[top].mean()),
        "recall_at_top": float(y[top].sum() / y.sum()),
        "lift_at_top": float(y[top].mean() / prevalence),
        "threshold": float(threshold), "flagged_share": float(yes.mean()),
        "precision": float(precision_score(y, yes, zero_division=0)),
        "recall": float(recall_score(y, yes, zero_division=0)),
        "f1": float(f1_score(y, yes, zero_division=0)),
        "accuracy": float(accuracy_score(y, yes)),
        "brier": float(brier_score_loss(y, s)) if in_range else float("nan"),
        "brier_baseline": prevalence * (1 - prevalence),
        "log_loss": float(log_loss(y, np.clip(s, 1e-6, 1 - 1e-6))) if in_range else float("nan"),
    }


# ----------------------------------------------------------------------------------------------
# risk bands
# ----------------------------------------------------------------------------------------------
def risk_bands(score, high: float = 0.10, medium: float = 0.20, index=None) -> pd.Series:
    """Turn scores into ``High`` (top ``high`` share), ``Medium`` (next ``medium`` share) and ``Low`` (the rest).

    Group sizes are exact: with 2,379 people and the defaults, 238 High, 476 Medium, 1,665 Low.
    Bands are based on RANK, so they do not depend on the score being a calibrated probability.
    People with identical scores at a boundary are separated by input order (arbitrary but deterministic).
    """
    s = np.asarray(score, dtype=float)
    if np.isnan(s).any():
        raise ValueError("score contains NaN.")
    if not (0 < high < 1 and 0 <= medium < 1 and high + medium < 1):
        raise ValueError("Need 0 < high < 1, 0 <= medium < 1 and high + medium < 1.")
    n = len(s)
    n_high = max(1, int(round(high * n)))
    n_med = int(round(medium * n))
    order = np.argsort(-s, kind="stable")
    band = np.full(n, "Low", dtype=object)
    band[order[:n_high]] = "High"
    band[order[n_high:n_high + n_med]] = "Medium"
    out = pd.Series(pd.Categorical(band, categories=list(BAND_LABELS), ordered=True), index=index)
    return out.rename("risk_band")


def band_summary(y_true, bands) -> pd.DataFrame:
    """People, leavers, leaver rate and lift per band, plus each band's share of all leavers."""
    y = pd.Series(np.asarray(y_true)).reset_index(drop=True)
    b = pd.Series(np.asarray(bands)).reset_index(drop=True)
    base = y.mean()
    rows = []
    for label in reversed(BAND_LABELS):
        m = b == label
        rows.append({"band": label, "people": int(m.sum()), "share_of_people": m.mean(),
                     "leavers": int(y[m].sum()), "leaver_rate": y[m].mean() if m.any() else np.nan,
                     "lift": y[m].mean() / base if m.any() else np.nan, "share_of_all_leavers": y[m].sum() / y.sum()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------------------------
def calibration_table(y_true, prob, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Group people by predicted probability; compare the average prediction with the observed leaver rate.

    ``quantile`` bins hold equal numbers of people (better when most scores are low); ``uniform`` bins are equal width.
    """
    y, p = _check_inputs(y_true, prob)
    if p.min() < 0 or p.max() > 1:
        raise ValueError("Calibration needs probabilities between 0 and 1.")
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'.")
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append({"bin": b + 1, "people": int(m.sum()), "mean_predicted": float(p[m].mean()),
                         "observed_rate": float(y[m].mean()), "leavers": int(y[m].sum())})
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """People-weighted average gap between predicted and observed rate (0 = perfectly calibrated)."""
    w = table["people"] / table["people"].sum()
    return float((w * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())


def calibrated_logreg(cv: int = 5) -> CalibratedClassifierCV:
    """Balanced logistic regression whose scores are mapped to probabilities (Platt scaling, fitted inside the training data)."""
    base = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv)


# ----------------------------------------------------------------------------------------------
# lift and gains
# ----------------------------------------------------------------------------------------------
def lift_table(y_true, score, n_bins: int = 10) -> pd.DataFrame:
    """Deciles (or n_bins groups) from highest to lowest score: leaver rate, lift, cumulative share of leavers caught."""
    y, s = _check_inputs(y_true, score)
    order = np.argsort(-s, kind="stable")
    groups = np.array_split(order, n_bins)
    base, total = y.mean(), y.sum()
    rows, cum = [], 0
    for i, g in enumerate(groups, 1):
        cum += y[g].sum()
        rows.append({"group": i, "people": len(g), "leavers": int(y[g].sum()), "leaver_rate": y[g].mean(),
                     "lift": y[g].mean() / base, "cumulative_share_of_leavers": cum / total,
                     "cumulative_share_of_people": sum(len(x) for x in groups[:i]) / len(y)})
    return pd.DataFrame(rows)


def gains_curve(y_true, score) -> tuple[np.ndarray, np.ndarray]:
    """Share of all leavers found (y) when reviewing the top x share of people, ranked by score."""
    y, s = _check_inputs(y_true, score)
    order = np.argsort(-s, kind="stable")
    caught = np.concatenate([[0], np.cumsum(y[order])]) / y.sum()
    reviewed = np.arange(len(y) + 1) / len(y)
    return reviewed, caught


def bootstrap_ci(stat, y_true, score, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for ``stat(y_true, score)``: how much it moves with a different sample of people."""
    y, s = _check_inputs(y_true, score)
    rng = np.random.default_rng(seed)
    vals = []
    while len(vals) < n_boot:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(stat(y[i], s[i]))
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
