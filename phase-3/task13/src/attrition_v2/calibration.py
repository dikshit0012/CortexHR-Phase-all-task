"""Phase 8: is a predicted 0.30 really a 30% chance? Measure first, calibrate only with out-of-fold evidence."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config
from cortexhr.evaluation import calibration_table, expected_calibration_error


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic regression of the outcome on logit(p). Slope < 1: over-confident; slope ~1 and intercept ~0: well calibrated."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def calibration_metrics(y, p, n_bins: int = 10) -> dict[str, float]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if p.min() < 0 or p.max() > 1:
        raise ValueError("Calibration metrics need probabilities between 0 and 1.")
    table = calibration_table(y, p, n_bins=n_bins, strategy="quantile")
    prev = y.mean()
    brier = brier_score_loss(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "brier": brier, "brier_reference_constant_prevalence": prev * (1 - prev),
        "brier_skill_vs_prevalence": 1 - brier / (prev * (1 - prev)),
        "ece_quantile_10bins": expected_calibration_error(table),
        "max_bin_gap": float((table["mean_predicted"] - table["observed_rate"]).abs().max()),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "calibration_slope": slope, "calibration_intercept": intercept,
        "mean_predicted": float(p.mean()), "observed_rate": float(prev),
        "p_min": float(p.min()), "p_median": float(np.median(p)), "p_max": float(p.max()),
        "roc_auc": roc_auc_score(y, p),
    }


def calibrated_oof(estimator_factory, X, y, folds, method: str, inner_splits: int = 5) -> np.ndarray:
    """Out-of-fold probabilities after calibration. In each outer fold the calibrator is fitted by an INNER cross-validation
    of the outer training part only (``CalibratedClassifierCV``), so the outer test rows never shape the calibration map."""
    y_arr = np.asarray(y)
    out = np.zeros(len(y_arr))
    for tr, te in folds:
        cal = CalibratedClassifierCV(
            estimator=estimator_factory(), method=method,
            cv=StratifiedKFold(inner_splits, shuffle=True, random_state=config.RANDOM_STATE + 2)).fit(X.iloc[tr], y_arr[tr])
        out[te] = cal.predict_proba(X.iloc[te])[:, 1]
    return out


def calibration_comparison(rows: dict[str, tuple[str, str, np.ndarray]], y) -> pd.DataFrame:
    """``rows``: label -> (model, method, out-of-fold probabilities). One metrics row each."""
    recs = []
    for label, (model, method, p) in rows.items():
        m = calibration_metrics(y, p)
        recs.append({"label": label, "model": model, "calibration": method, **m})
    return pd.DataFrame(recs)


def plot_calibration(y, curves: dict[str, np.ndarray], path, n_bins: int = 10) -> None:
    """Reliability diagram (quantile bins) + histogram of predicted probabilities."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, p in curves.items():
        t = calibration_table(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(t["mean_predicted"], t["observed_rate"], marker="o", lw=1.6, label=name)
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed leaver rate (bin)")
    ax.set_title("Reliability (out-of-fold, 10 equal-size bins)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.25)
    bins = np.linspace(0, 1, 26)
    for name, p in curves.items():
        ax2.hist(p, bins=bins, histtype="step", lw=1.5, label=name)
    ax2.set_xlabel("predicted probability"); ax2.set_ylabel("employees")
    ax2.set_title("Distribution of predicted probabilities"); ax2.legend(fontsize=8); ax2.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
