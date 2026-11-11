"""Independent recomputation of SHAP for a deterministic subset of employees (every STEP-th row), compared with reports/internal/shap_values.csv.
Uses a freshly built explainer (new background draw, same seeds). Writes reports/internal/shap_reproducibility.json."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from cortexhr_task14 import config
from cortexhr_task14.employee_scoring import build_scoring_inputs
from cortexhr_task14.model_loader import load_attrition_v2
from cortexhr_task14.shap_explainer import AttritionExplainer

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main() -> dict:
    loaded = load_attrition_v2()
    X = build_scoring_inputs(loaded).X
    cached = pd.read_csv(config.REPORTS / "internal" / "shap_values.csv", index_col=0)
    ex = AttritionExplainer(loaded, X)                       # fresh object: background re-drawn from the same seed
    rows = list(range(0, len(X), STEP))
    E = ex.encode(X)
    new = np.zeros((len(rows), E.shape[1]))
    for k, i in enumerate(rows):
        np.random.seed(config.SHAP_SEED + i)                 # identical to AttritionExplainer.explain
        new[k] = ex._exp.shap_values(E[i:i + 1], nsamples=ex.nsamples, l1_reg=False, silent=True)[0]
    old = cached.iloc[rows].to_numpy()
    top3 = lambda a: [frozenset(np.array(cached.columns)[np.argsort(-r)[:3]]) for r in a]
    res = {"employees_recomputed": len(rows), "of": len(X), "max_abs_difference": float(np.abs(new - old).max()),
           "identical_top3_sets": int(sum(a == b for a, b in zip(top3(new), top3(old)))), "base_value_new": ex.base_value,
           "settings": {"nsamples": ex.nsamples, "seed": ex.seed, "background_rows": len(ex.background_index)}}
    (config.REPORTS / "internal" / "shap_reproducibility.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
