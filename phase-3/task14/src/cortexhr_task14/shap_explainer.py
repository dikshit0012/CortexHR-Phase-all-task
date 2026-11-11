"""SHAP explanations of the attrition-v2 probability, per employee, on the ORIGINAL 33 input features.

Why KernelExplainer (and not Tree/Linear):
  attrition-v2 is NOT a tree model. It is CalibratedClassifierCV = 5 members, each = Task 2 pipeline (cleaning + imputation + scaling +
  one-hot) -> LogisticRegression -> sigmoid calibration, and p = mean of the 5 calibrated probabilities (reproduced exactly: max diff 0).
  * A LinearExplainer on the log-odds would explain a different function: sigmoid(mean of the 5 margins) differs from the real prediction by
    up to 0.08 on this data.
  * Explaining one-hot columns separately would fragment e.g. 'department' into many pieces and ignore the cleaner's cross-column rules.
  So the explainer wraps the COMPLETE fitted model (`predict_proba[:, 1]`, preprocessing included) and masks whole input features.

Scale: contributions are in PROBABILITY units (0-1; 0.02 = 2 percentage points of the 12-month attrition probability).
  prediction = base_value + sum(contributions)   (SHAP efficiency property, checked for every employee and reported).
  base_value = average prediction over the background sample (a reference group of active employees). Positive = raises the predicted risk
  relative to that reference, negative = lowers it. This describes the MODEL's behaviour; it is not causal evidence.
Method: Kernel SHAP (sampling approximation, seeded per employee). A model-agnostic cross-check of top-3 stability is in validation.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from . import config

MISSING = -1e9      # sentinel so that "missing" equals "missing" when SHAP compares an employee with the background


@dataclass
class ShapResult:
    values: pd.DataFrame                 # employees x 33 input features, probability units
    base_value: float
    predictions: np.ndarray              # attrition-v2 probabilities (same function that is explained)
    additivity_error: np.ndarray         # |base + sum(values) - prediction| per employee
    settings: Dict


class AttritionExplainer:
    def __init__(self, loaded, X: pd.DataFrame, n_background: int = config.SHAP_BACKGROUND_ROWS, nsamples: int = config.SHAP_NSAMPLES,
                 seed: int = config.SHAP_SEED):
        self.model, self.columns = loaded.model, list(X.columns)
        self.nsamples, self.seed = nsamples, seed
        self.cats = {c: sorted(X[c].dropna().astype(str).unique()) for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])}
        self.E = self.encode(X)
        n_bg = min(n_background, len(X))
        idx = np.random.RandomState(seed).choice(len(X), n_bg, replace=False)
        self.background_index = np.sort(idx)
        self._exp = shap.KernelExplainer(self._predict, self.E[self.background_index], link="identity")
        self.base_value = float(np.ravel(self._exp.expected_value)[0])

    # ---- encoding: numeric array <-> the raw DataFrame the saved pipeline expects --------------------------
    def encode(self, X: pd.DataFrame) -> np.ndarray:
        E = np.zeros(X.shape, float)
        for j, c in enumerate(self.columns):
            if c in self.cats:
                m = {v: i for i, v in enumerate(self.cats[c])}
                E[:, j] = [m.get(str(v), -1) if pd.notna(v) else -1 for v in X[c]]
            else:
                E[:, j] = X[c].astype(float).fillna(MISSING).to_numpy()
        return E

    def decode(self, E: np.ndarray) -> pd.DataFrame:
        E = np.atleast_2d(E)
        data = {}
        for j, c in enumerate(self.columns):
            if c in self.cats:
                lab = np.array(self.cats[c], dtype=object)
                idx = E[:, j].astype(int)
                data[c] = pd.Series([lab[i] if i >= 0 else np.nan for i in idx], dtype=object)
            else:
                v = E[:, j].copy()
                v[v <= MISSING / 2] = np.nan
                data[c] = v
        return pd.DataFrame(data)

    def _predict(self, E: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.decode(E))[:, 1]      # complete attrition-v2, positive class

    def explain(self, X: pd.DataFrame, progress: bool = False) -> ShapResult:
        E = self.encode(X)
        preds = self._predict(E)
        out = np.zeros(E.shape)
        for i in range(len(E)):
            np.random.seed(self.seed + i)                           # Kernel SHAP samples coalitions with the global RNG
            out[i] = self._exp.shap_values(E[i:i + 1], nsamples=self.nsamples, l1_reg=False, silent=True)[0]
            if progress and (i + 1) % 50 == 0:
                print(f"  SHAP {i + 1}/{len(E)}", flush=True)
        err = np.abs(self.base_value + out.sum(axis=1) - preds)
        return ShapResult(values=pd.DataFrame(out, columns=self.columns, index=X.index), base_value=self.base_value, predictions=preds,
                          additivity_error=err,
                          settings={"explainer": "shap.KernelExplainer", "output": "attrition-v2 predict_proba[:, 1] (probability units)",
                                    "background_rows": len(self.background_index), "nsamples": self.nsamples, "seed": self.seed,
                                    "features": len(self.columns), "feature_masking": "whole input features (pipeline-aware)"})
