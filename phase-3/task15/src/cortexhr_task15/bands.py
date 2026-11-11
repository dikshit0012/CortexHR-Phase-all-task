"""Salary bands: Mid = model prediction; Low / High = Mid x training out-of-fold quantiles of (actual / predicted).
Built ONLY from training-fold residuals; test labels are used afterwards to MEASURE coverage, never to construct the band.
It is an empirical prediction band with a nominal 80% level (10th-90th percentile), not a proven confidence interval.
Gender and career gap never enter (the band is a function of the job features only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class BandCalibration:
    low_ratio: float
    high_ratio: float
    n: int
    low_q: float = config.BAND_LOW_Q
    high_q: float = config.BAND_HIGH_Q


def fit_calibration(oof_pred, y_train, low_q: float = config.BAND_LOW_Q, high_q: float = config.BAND_HIGH_Q) -> BandCalibration:
    pred, y = np.asarray(oof_pred, float), np.asarray(y_train, float)
    ok = pred > 0
    if ok.sum() < 30:
        raise ValueError("too few training residuals to calibrate a band")
    ratio = y[ok] / pred[ok]
    return BandCalibration(float(np.quantile(ratio, low_q)), float(np.quantile(ratio, high_q)), int(ok.sum()), low_q, high_q)


def make_bands(mid, cal: BandCalibration) -> pd.DataFrame:
    """Round to whole rupees; guarantees 0 <= low <= mid <= high."""
    mid = np.maximum(np.asarray(mid, float), 0.0)
    low = np.minimum(mid * cal.low_ratio, mid)
    high = np.maximum(mid * cal.high_ratio, mid)
    return pd.DataFrame({"recommended_salary_low": np.round(low).astype(int), "predicted_salary_mid": np.round(mid).astype(int),
                         "recommended_salary_high": np.round(high).astype(int)})


def coverage(y, bands: pd.DataFrame) -> Dict[str, float]:
    y = np.asarray(y, float)
    inside = (y >= bands["recommended_salary_low"].to_numpy()) & (y <= bands["recommended_salary_high"].to_numpy())
    return {"coverage": float(inside.mean()), "below_low": float((y < bands["recommended_salary_low"].to_numpy()).mean()),
            "above_high": float((y > bands["recommended_salary_high"].to_numpy()).mean()),
            "mean_relative_width": float(((bands["recommended_salary_high"] - bands["recommended_salary_low"]) / bands["predicted_salary_mid"].clip(lower=1)).mean())}


def fit_level_calibration(oof_pred, y_train, levels, min_rows: int = 40, low_q: float = config.BAND_LOW_Q, high_q: float = config.BAND_HIGH_Q) -> Dict:
    """Band ratios per JOB LEVEL (relative error is larger at low levels, so one global ratio under-covers them). Levels with fewer than
    `min_rows` training residuals use the global ratios. Job level is a job attribute; no protected attribute is involved."""
    pred, y, lv = np.asarray(oof_pred, float), np.asarray(y_train, float), np.asarray(levels)
    glob = fit_calibration(pred, y, low_q, high_q)
    per = {int(l): (fit_calibration(pred[lv == l], y[lv == l], low_q, high_q) if (lv == l).sum() >= min_rows else glob) for l in np.unique(lv)}
    return {"global": glob, "by_level": per}


def make_bands_by_level(mid, levels, cals: Dict) -> pd.DataFrame:
    mid = np.maximum(np.asarray(mid, float), 0.0)
    low_r = np.array([cals["by_level"].get(int(l), cals["global"]).low_ratio for l in levels])
    high_r = np.array([cals["by_level"].get(int(l), cals["global"]).high_ratio for l in levels])
    low, high = np.minimum(mid * low_r, mid), np.maximum(mid * high_r, mid)
    return pd.DataFrame({"recommended_salary_low": np.round(low).astype(int), "predicted_salary_mid": np.round(mid).astype(int), "recommended_salary_high": np.round(high).astype(int)})
