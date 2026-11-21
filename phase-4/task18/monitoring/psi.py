"""Population Stability Index (PSI) for monitoring.

Why this exists instead of calling the course helper `psi()` in cortexhr_helpers.py (audited for Task 18; kept untouched):

  PSI = sum over bins of (current% - reference%) * ln(current% / reference%)

The helper implements that formula correctly for CONTINUOUS data without missing values, and this module reproduces it
bit-for-bit in that case (reference-quantile bins, current values clipped into the reference range, 1e-6 floor; verified by a
regression test against the helper's own source). The audit found it unsafe for the other cases a real workforce table contains:

  * low-cardinality / ordinal / binary columns: quantile edges collapse, so a 30% -> 60% shift in a 0/1 flag returned PSI = 0.0
    and a 1-4 rating shift whose exact PSI is 0.247 returned 0.0003 (a silent false "OK");
  * missing values: NaN in the reference silently returned 0.0, NaN in the current set shrank the denominators wrongly;
  * string categories (e.g. work_mode) raised ValueError;
  * an empty current set produced NaN with a numpy warning.

Differences from the helper, all deliberate:
  * numeric features with <= MAX_DISCRETE distinct reference values are compared value-by-value;
  * missing values form their own bin (a rising missing rate IS drift: a broken pipeline looks exactly like that);
  * categorical features use one bin per value over the UNION of reference and current categories, so a brand-new category
    is penalised through the zero-frequency floor instead of being ignored;
  * invalid input (non-numeric text in a numeric column, infinities, empty data, an un-binnable reference) raises PSIError
    and is reported as an error - it never turns into 0.0.

PSI measures a change in the distribution of an input. It does not measure model accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import monitoring_config as cfg

MISSING_LABEL = "<missing>"


class PSIError(ValueError):
    """The PSI could not be computed from the supplied data (never silently converted to 0)."""


@dataclass
class PSIResult:
    psi: float
    method: str                       # numeric_quantile | numeric_discrete | categorical
    n_bins: int
    reference_n: int
    current_n: int
    reference_missing: int
    current_missing: int
    top_bin: str                      # the bin contributing most to the PSI (aggregate label, e.g. "[2, 4)" or "Remote")
    top_bin_contribution: float
    top_bin_reference_pct: float
    top_bin_current_pct: float


def classify_psi(value: Optional[float], watch: float = cfg.PSI_WATCH, alert: float = cfg.PSI_ALERT) -> str:
    """OK if PSI < watch; WATCH if watch <= PSI <= alert; ALERT if PSI > alert; NOT_EVALUATED if PSI is None/NaN/inf.

    The value is rounded to PSI_DECIMALS first so the verdict always matches the number that is printed."""
    if value is None:
        return "NOT_EVALUATED"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "NOT_EVALUATED"
    if not math.isfinite(v):
        return "NOT_EVALUATED"
    v = round(v, cfg.PSI_DECIMALS)
    if v < watch:
        return "OK"
    if v <= alert:
        return "WATCH"
    return "ALERT"


# ------------------------------------------------------------------------------------------------ helpers
def _contributions(ref_counts: np.ndarray, cur_counts: np.ndarray, eps: float):
    ref_total, cur_total = float(ref_counts.sum()), float(cur_counts.sum())
    if ref_total <= 0 or cur_total <= 0:
        raise PSIError("a dataset has no rows")
    pr = np.clip(ref_counts / ref_total, eps, None)      # zero-frequency protection
    pc = np.clip(cur_counts / cur_total, eps, None)
    return (pc - pr) * np.log(pc / pr), pr, pc


def _finish(contrib, pr, pc, labels: Sequence[str], method: str, r_n: int, c_n: int, r_miss: int, c_miss: int) -> PSIResult:
    value = float(np.sum(contrib))
    if not math.isfinite(value):
        raise PSIError("PSI is not finite")
    k = int(np.argmax(contrib))
    return PSIResult(psi=round(value, cfg.PSI_DECIMALS), method=method, n_bins=len(labels), reference_n=r_n, current_n=c_n,
                     reference_missing=r_miss, current_missing=c_miss, top_bin=str(labels[k]),
                     top_bin_contribution=round(float(contrib[k]), cfg.PSI_DECIMALS),
                     top_bin_reference_pct=round(float(pr[k]) * 100, 2), top_bin_current_pct=round(float(pc[k]) * 100, 2))


def _to_float_array(values: Iterable[Any], what: str) -> np.ndarray:
    """Numeric column -> float array with NaN for missing. Non-numeric text or infinities are errors, not 'missing'."""
    s = pd.Series(list(values) if not isinstance(values, (pd.Series, np.ndarray)) else values, dtype=object)
    s = s.map(lambda v: np.nan if (isinstance(v, str) and not v.strip()) else v)         # blank text = missing
    num = pd.to_numeric(s, errors="coerce")
    bad = num.isna() & s.notna()
    if bad.any():
        raise PSIError(f"{what}: {int(bad.sum())} non-numeric value(s) in a numeric feature")
    arr = num.to_numpy(dtype=float)
    if np.isinf(arr).any():
        raise PSIError(f"{what}: infinite values")
    return arr


def _canonical(values: Iterable[Any]) -> List[str]:
    """Stable string label per value: 3, 3.0 and '3' are the same level; surrounding spaces are ignored; missing -> MISSING_LABEL.
    Case is NOT folded: 'remote' vs 'Remote' is a real upstream change that should be seen."""
    out: List[str] = []
    for v in (values.tolist() if isinstance(values, (pd.Series, np.ndarray)) else list(values)):
        if v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and math.isnan(v)):
            out.append(MISSING_LABEL)
            continue
        if isinstance(v, (bool, np.bool_)):
            out.append(str(bool(v)))
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            x = float(v)
            out.append(str(int(x)) if math.isfinite(x) and x.is_integer() else repr(x))
            continue
        t = str(v).strip()
        if not t:
            out.append(MISSING_LABEL)
            continue
        try:
            x = float(t)
            out.append(str(int(x)) if math.isfinite(x) and x.is_integer() else (repr(x) if math.isfinite(x) else t))
        except ValueError:
            out.append(t)
    return out


def _order(labels: Iterable[str]) -> List[str]:
    def key(s: str):
        try:
            return (0, float(s), s)
        except ValueError:
            return (1, 0.0, s) if s != MISSING_LABEL else (2, 0.0, s)
    return sorted(set(labels), key=key)


# ------------------------------------------------------------------------------------------------ public API
def psi_categorical(reference: Iterable[Any], current: Iterable[Any], eps: float = cfg.PSI_EPS, _method: str = "categorical") -> PSIResult:
    ref, cur = _canonical(reference), _canonical(current)
    if not ref or not cur:
        raise PSIError("a dataset has no rows")
    cats = _order(ref + cur)
    rc, cc = pd.Series(ref).value_counts(), pd.Series(cur).value_counts()
    rcounts = np.array([rc.get(k, 0) for k in cats], float)
    ccounts = np.array([cc.get(k, 0) for k in cats], float)
    contrib, pr, pc = _contributions(rcounts, ccounts, eps)
    return _finish(contrib, pr, pc, cats, _method, len(ref), len(cur), ref.count(MISSING_LABEL), cur.count(MISSING_LABEL))


def psi_numeric(reference: Iterable[Any], current: Iterable[Any], bins: int = cfg.PSI_BINS, eps: float = cfg.PSI_EPS,
                max_discrete: int = cfg.MAX_DISCRETE) -> PSIResult:
    r, c = _to_float_array(reference, "reference"), _to_float_array(current, "current")
    if r.size == 0 or c.size == 0:
        raise PSIError("a dataset has no rows")
    r_obs, c_obs = r[~np.isnan(r)], c[~np.isnan(c)]
    if r_obs.size == 0:
        raise PSIError("the reference has no observed values")
    if np.unique(r_obs).size <= max_discrete:                       # binary / ordinal / counts: compare level by level
        return psi_categorical(r, c, eps, _method="numeric_discrete")
    edges = np.unique(np.quantile(r_obs, np.linspace(0, 1, bins + 1)))          # reference-quantile edges, as in the helper
    if edges.size < 3:
        raise PSIError("the reference is too concentrated to form bins; treat this feature as categorical")
    ref_counts = np.append(np.histogram(r_obs, edges)[0], np.isnan(r).sum()).astype(float)
    cur_counts = np.append(np.histogram(np.clip(c_obs, edges[0], edges[-1]), edges)[0], np.isnan(c).sum()).astype(float)
    labels = [f"[{edges[i]:.6g}, {edges[i + 1]:.6g}{']' if i == len(edges) - 2 else ')'}" for i in range(len(edges) - 1)] + [MISSING_LABEL]
    contrib, pr, pc = _contributions(ref_counts, cur_counts, eps)
    return _finish(contrib, pr, pc, labels, "numeric_quantile", int(r.size), int(c.size), int(np.isnan(r).sum()), int(np.isnan(c).sum()))


def compute_psi(reference: Iterable[Any], current: Iterable[Any], kind: str) -> PSIResult:
    if kind == "numeric":
        return psi_numeric(reference, current)
    if kind == "categorical":
        return psi_categorical(reference, current)
    raise PSIError(f"unknown feature kind {kind!r}")
