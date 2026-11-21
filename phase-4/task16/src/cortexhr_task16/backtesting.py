"""Metrics and the six-month hold-out backtest (no information from the hidden months is used anywhere in fitting)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import advanced_forecaster as af
from . import baseline_forecaster as bf
from . import config, exits_forecaster as ex
from .event_handler import build_calendar

log = logging.getLogger(__name__)


def metrics(actual, forecast, lower=None, upper=None) -> dict:
    """MAE, RMSE, MAPE, WAPE, bias (mean forecast - actual; positive = over-forecast), interval coverage and width."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    e = f - a
    out = {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAPE_pct": float(np.mean(np.abs(e[a != 0] / a[a != 0])) * 100) if (a != 0).any() else np.nan,
        "WAPE_pct": float(np.abs(e).sum() / np.abs(a).sum() * 100) if np.abs(a).sum() > 0 else np.nan,
        "bias": float(np.mean(e)),
        "n": int(len(a)),
    }
    if lower is not None and upper is not None:
        lo, up = np.asarray(lower, float), np.asarray(upper, float)
        inside = (a >= lo) & (a <= up)
        out.update({"coverage_80_pct": float(inside.mean() * 100), "inside_80": int(inside.sum()), "mean_width_80": float(np.mean(up - lo))})
    else:
        out.update({"coverage_80_pct": np.nan, "inside_80": np.nan, "mean_width_80": np.nan})
    return out


def split_holdout(y: pd.Series, h: int = config.HORIZON) -> tuple[pd.Series, pd.Series]:
    y = y.sort_index()
    if len(y) <= h + config.CV_MIN_TRAIN:
        raise ValueError(f"Series of {len(y)} months is too short for a {h}-month hold-out")
    return y.iloc[:-h], y.iloc[-h:]


def backtest_series(y: pd.Series, cal: pd.DataFrame, model: str | None = None, h: int = config.HORIZON) -> dict:
    """Hide the last h months; select (unless `model` is given) and fit on the earlier months only; forecast the hidden months."""
    train, test = split_holdout(y, h)
    cal_train = cal.loc[train.index]
    cv = None
    if model is None:
        cv = af.cross_validate(train, cal_train, h)
        model = af.select_model(cv)
    base = bf.forecast(train, h)
    adv = af.fit_forecast(train, cal_train, model, h)
    res = pd.DataFrame({
        "month": test.index, "actual": test.values,
        "baseline_forecast": base["mean"].values, "baseline_lower_80": base["lower"].values, "baseline_upper_80": base["upper"].values,
        "advanced_forecast": adv["mean"].values, "advanced_lower_80": adv["lower"].values, "advanced_upper_80": adv["upper"].values,
    })
    res["baseline_abs_error"] = (res["baseline_forecast"] - res["actual"]).abs()
    res["advanced_abs_error"] = (res["advanced_forecast"] - res["actual"]).abs()
    return {"model": model, "cv": cv, "results": res, "train_end": train.index[-1], "n_train": len(train)}


def holdout_candidates(y: pd.Series, cal: pd.DataFrame, h: int = config.HORIZON) -> pd.DataFrame:
    """Post-hoc view of every candidate on the hold-out. Shown for transparency ONLY; it is not used to choose a model."""
    train, test = split_holdout(y, h)
    rows = []
    for spec in af.CANDIDATES:
        fc = af.run_candidate(spec, train, cal.loc[train.index], h)
        m = metrics(test.values, fc["mean"].values, fc["lower"].values, fc["upper"].values)
        rows.append({"model": spec.name, **m})
    return pd.DataFrame(rows)


def backtest_exits(rate: pd.Series, exits: pd.Series, headcount: pd.Series, headcount_fc_mean: np.ndarray, h: int = config.HORIZON) -> dict:
    """Backtest the attrition-rate forecast and the implied exit counts.

    Methods: the one selected by CV on the training months, and the sample file's same-month-3y average (reference).
    Exit counts = forecast rate x opening headcount, where opening headcount comes from the advanced headcount forecast.
    """
    train, test = rate.iloc[:-h], rate.iloc[-h:]
    cv = ex.cv_methods(train, h)
    chosen = ex.select_method(cv)
    opening = np.r_[headcount.iloc[-h - 1], np.asarray(headcount_fc_mean, float)[:-1]]
    actual_exits = exits.iloc[-h:].values.astype(float)
    rows = []
    for name in dict.fromkeys([chosen, ex.REFERENCE_METHOD]):
        fc = ex.forecast_rate(train, name, h)
        counts = pd.DataFrame({"month": test.index, "actual_exits": actual_exits,
                               "expected_exits": fc["mean"].values / 100 * opening,
                               "lower_80": fc["lower"].values / 100 * opening, "upper_80": fc["upper"].values / 100 * opening})
        rows.append({"method": name, "selected": name == chosen, "rate": fc,
                     "metrics_rate": metrics(test.values, fc["mean"].values, fc["lower"].values, fc["upper"].values),
                     "counts": counts,
                     "metrics_count": metrics(actual_exits, counts["expected_exits"], counts["lower_80"], counts["upper_80"])})
    return {"selected": chosen, "cv": cv, "rows": rows, "actual_rate": test}
