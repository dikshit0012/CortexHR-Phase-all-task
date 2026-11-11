"""Informational check: are the top-3 drivers stable when SHAP is recomputed with 4x more sampled coalitions?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .shap_explainer import AttritionExplainer


def stability_check(loaded, X: pd.DataFrame, shap_res, n: int = 12, nsamples: int = 6000) -> dict:
    idx = list(range(0, len(X), max(1, len(X) // n)))[:n]
    ex = AttritionExplainer(loaded, X, nsamples=nsamples)
    ref = ex.explain(X.iloc[idx]).values
    cur = shap_res.values.iloc[idx]
    top = lambda v: [frozenset(r.nlargest(3).index) for _, r in v.iterrows()]
    a, b = top(cur), top(ref)
    return {"employees": len(idx), "nsamples_reference": nsamples, "nsamples_used": int(shap_res.settings["nsamples"]),
            "identical_top3_sets": int(sum(x == y for x, y in zip(a, b))), "mean_top3_overlap": float(np.mean([len(x & y) / 3 for x, y in zip(a, b)])),
            "max_abs_difference": float(np.abs(cur.to_numpy() - ref.to_numpy()).max())}
