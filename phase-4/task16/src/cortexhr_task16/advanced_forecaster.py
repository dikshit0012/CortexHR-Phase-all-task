"""Advanced statistical forecasters (statsmodels / numpy) and leakage-free model selection.

Candidates (all fitted on history only):
  drift_events          net monthly change = constant + event dummies (OLS, normal errors)           complexity 1
  ets_damped            exponential smoothing, additive error, damped additive trend (ETSModel)       complexity 2
  drift_events_fourier  drift_events + 2 annual Fourier pairs (smooth seasonality in monthly change)  complexity 3
  ets_holt_winters      ETS with damped trend and additive 12-month seasonality                       complexity 4
  sarima                SARIMAX (0,1,1)(0,1,0,12)                                                     complexity 5

The model is picked by rolling-origin cross-validation INSIDE the training window; the simplest candidate whose CV MAE is within
SELECTION_TOLERANCE of the best wins, so complexity must earn its place. Hold-out data is never touched.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import baseline_forecaster, config
from .event_handler import EFFECT_DUMMIES
from .preprocessing import future_index

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ helpers
def _quantiles(alpha: float) -> float:
    return float(norm.ppf(1 - (1 - alpha) / 2))


def _frame(index, mean, lower, upper) -> pd.DataFrame:
    return pd.DataFrame({"mean": np.asarray(mean, float), "lower": np.asarray(lower, float), "upper": np.asarray(upper, float)}, index=index)


def _fourier(index: pd.DatetimeIndex, k: int) -> np.ndarray:
    m = index.month.values
    cols = []
    for j in range(1, k + 1):
        cols += [np.sin(2 * np.pi * j * m / 12), np.cos(2 * np.pi * j * m / 12)]
    return np.column_stack(cols) if cols else np.empty((len(index), 0))


# ------------------------------------------------------------------ candidates
def drift_events(y: pd.Series, cal_hist: pd.DataFrame, cal_future: pd.DataFrame, h: int, alpha: float, fourier_k: int = 0) -> pd.DataFrame:
    """OLS on the monthly net change; level forecast = last value + cumulated change.

    Variance of the k-step level forecast = k*sigma^2 + parameter uncertainty of the cumulated regressors, so the interval
    widens with the horizon and includes estimation error.
    """
    d = y.diff().dropna()
    idx_f = future_index(y.index[-1], h)
    keep = [c for c in EFFECT_DUMMIES if cal_hist.loc[d.index, c].std() > 0]      # a dummy never active in training cannot be estimated
    X = np.column_stack([np.ones(len(d)), cal_hist.loc[d.index, keep].values, _fourier(d.index, fourier_k)])
    Xf = np.column_stack([np.ones(h), cal_future.loc[idx_f, keep].values, _fourier(idx_f, fourier_k)])
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, d.values, rcond=None)
    resid = d.values - X @ beta
    sigma2 = float(resid @ resid / max(n - p, 1))
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    C = np.cumsum(Xf, axis=0)
    mean = y.values[-1] + C @ beta
    var = np.einsum("ij,jk,ik->i", C, cov, C) + sigma2 * np.arange(1, h + 1)
    z = _quantiles(alpha)
    sd = np.sqrt(var)
    return _frame(idx_f, mean, mean - z * sd, mean + z * sd)


def _ets(y: pd.Series, h: int, alpha: float, seasonal: bool) -> pd.DataFrame:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    model = ETSModel(y, error="add", trend="add", damped_trend=True,
                     seasonal="add" if seasonal else None, seasonal_periods=config.SEASONAL_PERIOD if seasonal else None)
    res = model.fit(disp=False, maxiter=200)
    pred = res.get_prediction(start=len(y), end=len(y) + h - 1).summary_frame(alpha=1 - alpha)
    return _frame(future_index(y.index[-1], h), pred["mean"], pred["pi_lower"], pred["pi_upper"])


def _sarima(y: pd.Series, h: int, alpha: float) -> pd.DataFrame:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    res = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 0, 12), enforce_invertibility=False).fit(disp=False, maxiter=200)
    pred = res.get_forecast(h).summary_frame(alpha=1 - alpha)
    return _frame(future_index(y.index[-1], h), pred["mean"], pred["mean_ci_lower"], pred["mean_ci_upper"])


@dataclass(frozen=True)
class Spec:
    name: str
    label: str
    complexity: int
    run: Callable[[pd.Series, pd.DataFrame, pd.DataFrame, int, float], pd.DataFrame]


CANDIDATES: list[Spec] = [
    Spec("drift_events", "Drift + event dummies (OLS on monthly change)", 1,
         lambda y, ch, cf, h, a: drift_events(y, ch, cf, h, a, 0)),
    Spec("ets_damped", "ETS(A,Ad,N) damped trend", 2, lambda y, ch, cf, h, a: _ets(y, h, a, False)),
    Spec("drift_events_fourier", "Drift + event dummies + 2 seasonal harmonics", 3,
         lambda y, ch, cf, h, a: drift_events(y, ch, cf, h, a, 2)),
    Spec("ets_holt_winters", "ETS(A,Ad,A) damped trend + additive seasonality", 4, lambda y, ch, cf, h, a: _ets(y, h, a, True)),
    Spec("sarima", "SARIMA(0,1,1)(0,1,0)12", 5, lambda y, ch, cf, h, a: _sarima(y, h, a)),
]
SPECS = {s.name: s for s in CANDIDATES}


def future_calendar(cal_hist: pd.DataFrame, h: int) -> pd.DataFrame:
    """Calendar for forecast months: no known future workforce events (assumed adjustments are applied separately,
    after the model, so they can never be double counted with the dummies)."""
    idx = future_index(cal_hist.index[-1], h)
    return pd.DataFrame(0.0, index=idx, columns=cal_hist.columns)


def run_candidate(spec: Spec, y: pd.Series, cal_hist: pd.DataFrame, h: int = config.HORIZON, alpha: float = config.INTERVAL) -> pd.DataFrame:
    """Fit on y only, forecast h months, clip to non-negative headcount."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = spec.run(y, cal_hist.loc[y.index], future_calendar(cal_hist.loc[y.index], h), h, alpha)
    fc = fc.clip(lower=0.0)
    fc["lower"] = np.minimum(fc["lower"], fc["mean"])
    fc["upper"] = np.maximum(fc["upper"], fc["mean"])
    return fc


# ------------------------------------------------------------------ selection
def cross_validate(y: pd.Series, cal: pd.DataFrame, h: int = config.HORIZON, min_train: int = config.CV_MIN_TRAIN,
                   step: int = config.CV_STEP, alpha: float = config.INTERVAL, specs: list[Spec] | None = None) -> pd.DataFrame:
    """Rolling-origin CV using only observations in y. Returns one row per model (plus the seasonal baseline)."""
    specs = specs or CANDIDATES
    origins = list(range(min_train, len(y) - h + 1, step))
    if not origins:
        raise ValueError(f"Series too short for cross-validation ({len(y)} months; need >= {min_train + h})")
    rows = []
    for spec in specs:
        abs_err, covered, fails = [], [], 0
        for o in origins:
            try:
                fc = run_candidate(spec, y.iloc[:o], cal.iloc[:o], h, alpha)
            except Exception as exc:                          # a failed fit disqualifies that origin, loudly
                log.warning("CV fit failed for %s at origin %d: %s", spec.name, o, exc)
                fails += 1
                continue
            a = y.iloc[o:o + h].values
            abs_err += list(np.abs(a - fc["mean"].values)); covered += list((a >= fc["lower"].values) & (a <= fc["upper"].values))
        rows.append({"model": spec.name, "label": spec.label, "complexity": spec.complexity, "cv_mae": np.mean(abs_err) if abs_err else np.inf,
                     "cv_coverage_80": np.mean(covered) if covered else np.nan, "n_origins": len(origins), "failed_fits": fails})
    base_err = []
    for o in origins:
        f = baseline_forecaster.point_forecast(y.iloc[:o], h)
        base_err += list(np.abs(y.iloc[o:o + h].values - f.values))
    rows.append({"model": "seasonal_baseline", "label": "Seasonal baseline (reference only)", "complexity": 0, "cv_mae": float(np.mean(base_err)),
                 "cv_coverage_80": np.nan, "n_origins": len(origins), "failed_fits": 0})
    return pd.DataFrame(rows)


def select_model(cv: pd.DataFrame, tolerance: float = config.SELECTION_TOLERANCE) -> str:
    """Simplest candidate whose CV MAE is within `tolerance` of the best candidate (the baseline is not eligible)."""
    cand = cv[cv["model"] != "seasonal_baseline"].dropna(subset=["cv_mae"])
    best = cand["cv_mae"].min()
    eligible = cand[cand["cv_mae"] <= best * (1 + tolerance)]
    return str(eligible.sort_values(["complexity", "cv_mae"]).iloc[0]["model"])


def fit_forecast(y: pd.Series, cal_hist: pd.DataFrame, model: str, h: int = config.HORIZON, alpha: float = config.INTERVAL) -> pd.DataFrame:
    return run_candidate(SPECS[model], y, cal_hist, h, alpha)


def continuity_check(y: pd.Series, fc: pd.DataFrame) -> dict:
    """Does the forecast stay inside the workforce's historical range of month-to-month movement?"""
    steps = np.abs(np.diff(np.r_[y.values[-1], fc["mean"].values]))
    hist_max = float(np.abs(y.diff().dropna()).max())
    return {"max_forecast_step": float(steps.max()), "max_historical_step": hist_max, "ok": bool(steps.max() <= hist_max)}
