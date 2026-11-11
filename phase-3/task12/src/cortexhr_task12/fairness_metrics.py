"""Twin-gap statistics. Scale: the Task 11 final_score, bounded in [0, 1].

Twin Gap = |score_A - score_B|.  Relative gap = gap / mean(score_A, score_B) (undefined when both scores are 0).
Tolerance (documented): 0.01 = one point on a 0-100 view of the score. `non-zero` uses a numerical epsilon of 1e-9.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .cortexhr_helpers import group_gap as _helper_group_gap   # the course-provided helper, used unmodified

TOLERANCE = 0.01
ZERO_EPS = 1e-9


def twin_gap(a: float, b: float) -> float:
    return abs(float(a) - float(b))


def relative_gap(a: float, b: float) -> float:
    m = (float(a) + float(b)) / 2.0
    return float("nan") if m <= 0 else abs(float(a) - float(b)) / m


def gap_summary(gaps: Iterable[float], tolerance: float = TOLERANCE) -> Dict[str, float]:
    g = np.asarray(list(gaps), float)
    if g.size == 0 or np.isnan(g).any():
        raise ValueError("gaps must be a non-empty, NaN-free sequence")
    return {"n_pairs": int(g.size), "average_gap": float(g.mean()), "median_gap": float(np.median(g)), "maximum_gap": float(g.max()),
            "minimum_gap": float(g.min()), "non_zero_gap_pairs": int((g > ZERO_EPS).sum()),
            "pairs_exceeding_tolerance": int((g > tolerance).sum()), "tolerance": tolerance}


def reduction_pct(before: float, after: float) -> float:
    """((before - after) / before) * 100; NaN when the baseline is 0 (nothing to reduce), never a division error."""
    if before is None or after is None or not math.isfinite(before) or not math.isfinite(after) or abs(before) < 1e-15:
        return float("nan")
    return (before - after) / before * 100.0


def compare_summaries(before: Dict[str, float], after: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for key, label in [("average_gap", "Average twin gap"), ("median_gap", "Median twin gap"), ("maximum_gap", "Maximum twin gap"),
                       ("minimum_gap", "Minimum twin gap"), ("non_zero_gap_pairs", "Non-zero gap pairs"),
                       ("pairs_exceeding_tolerance", f"Pairs above tolerance ({before['tolerance']})")]:
        b, a = before[key], after[key]
        rows.append({"metric": label, "before": b, "after": a, "improvement_absolute": b - a, "improvement_pct": reduction_pct(b, a)})
    return pd.DataFrame(rows)


def group_gap_report(scores, groups) -> Dict:
    """Course helper `group_gap`: mean score per group and the largest gap between group means. A warning signal, not proof."""
    means, gap = _helper_group_gap(scores, groups)
    return {"group_means": means, "max_gap": gap}
