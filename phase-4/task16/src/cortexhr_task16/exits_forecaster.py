"""Expected employee exits, forecast as the monthly attrition rate (exits / headcount at start of month, in %).

The rate is forecast directly because it is what sample_forecast asks for (metric attrition_rate_monthly_pct) and because exit
COUNTS scale with headcount. Expected exits = rate x projected opening headcount. Hires are NOT forecast separately; the
"implied hires" figure is derived from the headcount forecast (net change + expected exits) and is labelled as such.

Candidates (chosen by rolling-origin CV inside the training window): trailing mean of 12/24/36 months, and the same-month
average of the last three years (the method used by the sample file). Intervals are empirical 10th/90th percentiles of that
method's own past errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .preprocessing import future_index


def _trailing(window: int):
    return lambda r, h: np.repeat(r.values[-window:].mean(), h)


def _same_month_3y(r: pd.Series, h: int) -> np.ndarray:
    out = []
    idx = future_index(r.index[-1], h)
    for t in idx:
        vals = [r.get(t - pd.DateOffset(years=k)) for k in (1, 2, 3)]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        out.append(np.mean(vals) if vals else r.values[-12:].mean())
    return np.array(out)


METHODS = {
    "trailing_12m_mean": ("mean of the last 12 months", 1, _trailing(12)),
    "trailing_24m_mean": ("mean of the last 24 months", 2, _trailing(24)),
    "trailing_36m_mean": ("mean of the last 36 months", 3, _trailing(36)),
    "same_month_3y_avg": ("same-month average of the last 3 years", 4, _same_month_3y),
}
REFERENCE_METHOD = "same_month_3y_avg"        # the sample_forecast baseline


def predict(r: pd.Series, method: str, h: int = config.HORIZON) -> np.ndarray:
    return np.maximum(METHODS[method][2](r, h), 0.0)


def cv_methods(r: pd.Series, h: int = config.HORIZON, min_train: int = config.CV_MIN_TRAIN, step: int = 3) -> pd.DataFrame:
    rows = []
    origins = list(range(min_train, len(r) - h + 1, step))
    for name, (label, cx, _) in METHODS.items():
        err = []
        for o in origins:
            err += list(np.abs(r.iloc[o:o + h].values - predict(r.iloc[:o], name, h)))
        rows.append({"method": name, "label": label, "complexity": cx, "cv_mae_pct_points": float(np.mean(err)), "n_origins": len(origins)})
    return pd.DataFrame(rows)


def select_method(cv: pd.DataFrame, tolerance: float = config.SELECTION_TOLERANCE) -> str:
    best = cv["cv_mae_pct_points"].min()
    ok = cv[cv["cv_mae_pct_points"] <= best * (1 + tolerance)]
    return str(ok.sort_values(["complexity", "cv_mae_pct_points"]).iloc[0]["method"])


def residual_quantiles(r: pd.Series, method: str, h: int = config.HORIZON, alpha: float = config.INTERVAL,
                       min_train: int = config.CV_MIN_TRAIN) -> tuple[float, float]:
    err = []
    for o in range(min_train, len(r) - 1):
        a = r.iloc[o:o + h].values
        err += list(a - predict(r.iloc[:o], method, h)[:len(a)])
    lo, hi = (1 - alpha) / 2, 1 - (1 - alpha) / 2
    return float(np.quantile(err, lo)), float(np.quantile(err, hi))


def forecast_rate(r: pd.Series, method: str, h: int = config.HORIZON, alpha: float = config.INTERVAL) -> pd.DataFrame:
    point = predict(r, method, h)
    lo, hi = residual_quantiles(r, method, h, alpha)
    out = pd.DataFrame({"mean": point, "lower": np.maximum(point + lo, 0.0), "upper": point + hi}, index=future_index(r.index[-1], h))
    return out


def expected_exits(rate_fc: pd.DataFrame, headcount_fc: pd.DataFrame, last_headcount: float) -> pd.DataFrame:
    """Convert a rate forecast to exit counts using the projected opening headcount of each month."""
    opening = np.r_[last_headcount, headcount_fc["mean"].values[:-1]]
    out = pd.DataFrame(index=rate_fc.index)
    out["opening_headcount"] = opening
    out["expected_attrition_rate_pct"] = rate_fc["mean"].values
    for c in ("mean", "lower", "upper"):
        out[{"mean": "expected_exits", "lower": "lower_80", "upper": "upper_80"}[c]] = rate_fc[c].values / 100.0 * opening
    out["expected_net_change"] = headcount_fc["mean"].values - opening
    out["implied_hires_needed"] = out["expected_net_change"] + out["expected_exits"]       # derived, not separately modelled
    return out
