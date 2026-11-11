"""Risk bands from the OFFICIAL helper `risk_bands()` (cortexhr_helpers.py, copied verbatim). No thresholds are defined here.

Official rule (docstring of the helper): High = top 10% of scores, Medium = next 20%, Low = the rest. It is RANK-BASED: the cut-offs are the
90th and 70th percentiles of the scores passed in, so bands describe an employee's position in THIS batch, not an absolute probability."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .cortexhr_helpers import risk_bands as official_risk_bands
from .employee_scoring import ScoringDataError, validate_probabilities

VALID_BANDS = ("Low", "Medium", "High")


def assign_risk_bands(probabilities) -> pd.Series:
    """Band for every employee. NaN is rejected first: the official helper would silently label NaN as 'Low'."""
    s = pd.Series(probabilities, dtype=float)
    validate_probabilities(s.to_numpy())
    return pd.Series(official_risk_bands(s.to_numpy()), index=s.index, dtype="object")


def band_thresholds(probabilities) -> Dict[str, float]:
    """The cut-offs the helper applies (its own 0.90 / 0.70 quantiles), reported for the README. Not an independent definition."""
    s = pd.Series(probabilities, dtype=float)
    return {"high_cutoff_p90": float(s.quantile(0.90)), "medium_cutoff_p70": float(s.quantile(0.70))}


def verify_bands(probabilities, bands) -> Dict:
    p, b = pd.Series(probabilities, dtype=float).reset_index(drop=True), pd.Series(bands).reset_index(drop=True)
    if len(p) != len(b) or b.isna().any() or not set(b) <= set(VALID_BANDS):
        raise ScoringDataError("invalid or missing risk bands")
    t = band_thresholds(p)
    ok_high = bool((p[b == "High"] >= t["high_cutoff_p90"]).all())
    ok_med = bool(((p[b == "Medium"] >= t["medium_cutoff_p70"]) & (p[b == "Medium"] < t["high_cutoff_p90"])).all())
    ok_low = bool((p[b == "Low"] < t["medium_cutoff_p70"]).all())
    if not (ok_high and ok_med and ok_low):
        raise ScoringDataError("a risk band does not match its probability")
    return {"counts": b.value_counts().reindex(VALID_BANDS).fillna(0).astype(int).to_dict(), **t,
            "min_score_by_band": p.groupby(b).min().to_dict(), "max_score_by_band": p.groupby(b).max().to_dict(), "consistent": True}
