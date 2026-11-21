"""Seasonal baseline: same month last year + estimated annual trend, with empirical prediction intervals.

    forecast(t) = headcount(t - 12k) + k * annual_trend,   k = ceil(step / 12)

annual_trend = mean year-over-year change over the last 12 months available AT THE FORECAST ORIGIN (needs 24 months);
with 13-23 months it is the single latest year-over-year change; with fewer than 13 it falls back to last value + monthly drift.
Intervals use the 10th/90th percentiles of this method's own past errors (rolling origins inside the training data only).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config
from .preprocessing import future_index


def annual_trend(y: pd.Series, period: int = config.SEASONAL_PERIOD) -> float:
    n = len(y)
    if n >= 2 * period:
        yoy = y.values[-period:] - y.values[-2 * period:-period]
        return float(np.mean(yoy))
    if n >= period + 1:
        return float(y.values[-1] - y.values[-1 - period])
    return float(period * (y.values[-1] - y.values[0]) / max(n - 1, 1))


def point_forecast(y: pd.Series, horizon: int = config.HORIZON, period: int = config.SEASONAL_PERIOD) -> pd.Series:
    n = len(y)
    idx = future_index(y.index[-1], horizon)
    trend = annual_trend(y, period)
    out = []
    for k in range(1, horizon + 1):
        cycles = math.ceil(k / period)
        pos = n - 1 + k - cycles * period
        if n < period + 1 or pos < 0:                       # too short for a seasonal reference: drift only
            out.append(y.values[-1] + k * trend / period)
        else:
            out.append(y.values[pos] + cycles * trend)
    return pd.Series(np.maximum(out, 0.0), index=idx, name="baseline")


def historical_errors(y: pd.Series, horizon: int = config.HORIZON, min_train: int = 24) -> pd.DataFrame:
    """Errors (actual - forecast) of the baseline made at every origin >= min_train inside `y` (no data outside y is used)."""
    rows = []
    for o in range(min(min_train, len(y) - 1), len(y)):
        f = point_forecast(y.iloc[:o], horizon)
        a = y.iloc[o:o + horizon]
        for step, (actual, fc) in enumerate(zip(a.values, f.values[:len(a)]), start=1):
            rows.append({"origin": o, "step": step, "error": actual - fc})
    return pd.DataFrame(rows, columns=["origin", "step", "error"])


def forecast(y: pd.Series, horizon: int = config.HORIZON, alpha: float = config.INTERVAL) -> pd.DataFrame:
    """Point forecast + empirical alpha-prediction interval. Columns: mean, lower, upper."""
    point = point_forecast(y, horizon)
    errs = historical_errors(y, horizon)
    lo_q, hi_q = (1 - alpha) / 2, 1 - (1 - alpha) / 2
    lowers, uppers = [], []
    pooled = errs["error"].values
    for step in range(1, horizon + 1):
        e = errs.loc[errs["step"] == step, "error"].values
        if len(e) < 8:
            e = pooled
        if len(e) < 8:                                       # very short series: random-walk spread
            sigma = float(np.std(np.diff(y.values))) if len(y) > 2 else 1.0
            from scipy.stats import norm
            z = norm.ppf(hi_q)
            lowers.append(-z * sigma * math.sqrt(step)); uppers.append(z * sigma * math.sqrt(step))
        else:
            lowers.append(float(np.quantile(e, lo_q))); uppers.append(float(np.quantile(e, hi_q)))
    out = pd.DataFrame({"mean": point.values, "lower": point.values + np.array(lowers), "upper": point.values + np.array(uppers)},
                       index=point.index)
    out[["lower", "upper"]] = out[["lower", "upper"]].clip(lower=0.0)
    out["lower"] = np.minimum(out["lower"], out["mean"]); out["upper"] = np.maximum(out["upper"], out["mean"])
    return out
