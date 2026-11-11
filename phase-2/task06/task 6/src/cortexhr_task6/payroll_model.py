"""Task 6 - Isolation Forest for payroll ratio features.

No labels are accepted anywhere in this module. `assert_no_forbidden_columns`
makes the "no identifiers / no label columns" rule a hard failure, not a promise.

Meaning of the outputs
  if_anomaly_score : higher = more abnormal (negated sklearn decision_function).
  if_flag          : 1 if the row's score is in the most abnormal `contamination`
                     share of the training rows (sklearn's own threshold), else 0.
                     It means "statistically unusual", NOT "proven payroll error".

Contamination (default 0.025): a prior for the share of unusual payslips, chosen
WITHOUT looking at labels. It matches the published `anomaly_baselines` payroll
recipe (2.5%), so the comparison changes the features, not the alert budget.
Set `score_threshold` to override the contamination cut-off with a fixed score.

Training: unsupervised and transductive - the forest is fitted on the same payroll
rows it scores (as the baseline and Task 5 did). No labels are involved, so there
is no label leakage; scores of the 18-month history are relative to each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from .payroll_features import EMPLOYEE_COL, FORBIDDEN_MODEL_COLUMNS, MODEL_FEATURES, MONTH_COL


@dataclass
class ModelConfig:
    contamination: float = 0.025
    n_estimators: int = 200
    random_state: int = 42
    score_threshold: Optional[float] = None   # if set, overrides contamination for if_flag
    # True (default): rows are put in a canonical order (month, employee, feature values) before fitting, so
    # the fitted forest - and every flag - is the same however the input rows were ordered.
    # False: fit in input-file order (ORIGINAL EXPERIMENT only; results change if rows are shuffled).
    canonical_row_order: bool = True
    features: List[str] = field(default_factory=lambda: list(MODEL_FEATURES))


def assert_no_forbidden_columns(features: List[str]) -> None:
    """Raise if an identifier or label-derived column is among the model features."""
    bad = [c for c in features if c in FORBIDDEN_MODEL_COLUMNS or c.startswith(("rule_", "label_"))]
    if bad:
        raise ValueError(f"forbidden columns in model features (identifier / label / rule): {bad}")


class PayrollAnomalyModel:
    """Median-impute -> IsolationForest. Preprocessing is fitted on the training rows only."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        assert_no_forbidden_columns(self.config.features)
        self.imputer_: Optional[SimpleImputer] = None
        self.model_: Optional[IsolationForest] = None
        self.kept_features_: List[str] = []
        self.dropped_constant_: List[str] = []

    def _matrix(self, frame: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def _canonical_index(self, features: pd.DataFrame, X: pd.DataFrame) -> pd.Index:
        """Index of the rows sorted by month, employee and feature values (never by id or file position)."""
        keys = pd.DataFrame(index=X.index)
        if MONTH_COL in features.columns:
            keys[MONTH_COL] = features[MONTH_COL]
        if EMPLOYEE_COL in features.columns:
            keys[EMPLOYEE_COL] = features[EMPLOYEE_COL]
        keys = keys.join(X)
        return keys.sort_values(list(keys.columns), kind="mergesort", na_position="last").index

    def fit(self, features: pd.DataFrame) -> "PayrollAnomalyModel":
        cols = self.config.features
        missing = [c for c in cols if c not in features.columns]
        if missing:
            raise ValueError(f"feature frame is missing model features: {missing}")
        X = self._matrix(features, cols)
        nunique = X.nunique(dropna=True)
        self.dropped_constant_ = [c for c in cols if nunique[c] <= 1]
        self.kept_features_ = [c for c in cols if c not in self.dropped_constant_]
        if not self.kept_features_:
            raise ValueError("no usable (non-constant) features to fit on")
        X = X[self.kept_features_]
        if self.config.canonical_row_order:
            X = X.loc[self._canonical_index(features, X)]
        # Missing values (undefined ratios) are imputed with the training median - a documented,
        # neutral choice that makes undefined ratios look "typical" (rules cover those cases).
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        self.model_ = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        ).fit(self.imputer_.transform(X))
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("call fit() before predict()")
        Xi = self.imputer_.transform(self._matrix(features, self.kept_features_))
        score = -self.model_.decision_function(Xi)       # higher = more abnormal
        if self.config.score_threshold is not None:
            flag = (score >= self.config.score_threshold).astype(int)
        else:
            flag = (self.model_.predict(Xi) == -1).astype(int)
        return pd.DataFrame({"if_flag": flag, "if_anomaly_score": score}, index=features.index)

    def fit_predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).predict(features)
