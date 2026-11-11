"""Phase 3 + 5: gradient-boosting model inside the existing Task 2 pipeline, plus the reference models."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from . import config
from cortexhr.baselines import model_pipeline   # Pipeline([("features", Task 2 pipeline), ("model", clf)])

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Starting point suggested in the task brief. It is a REFERENCE run (E00), not an assumption that it is right:
# with 349 leavers (not millions of rows) the tuning grid below explores shallower, more regularised trees.
BRIEF_DEFAULTS = dict(n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=15, min_child_samples=20,
                      subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0)


def make_lgbm(params: dict | None = None, class_weight: str | None = "balanced") -> LGBMClassifier:
    """LightGBM with a fixed seed. ``class_weight='balanced'`` matches Task 3's balanced baselines so the yes/no
    metrics at the 0.5 cut-off are comparable; calibration (Phase 8) restores honest probabilities."""
    p = dict(BRIEF_DEFAULTS)
    p.update(params or {})
    return LGBMClassifier(**p, subsample_freq=1 if p["subsample"] < 1 else 0, class_weight=class_weight,
                          random_state=config.RANDOM_STATE, n_jobs=1, deterministic=True, force_row_wise=True,
                          verbose=-1)


def v2_pipeline(params: dict | None = None, class_weight: str | None = "balanced"):
    """Task 2 feature pipeline (re-fitted inside every fold) + LightGBM, as ONE estimator."""
    return model_pipeline(make_lgbm(params, class_weight))


def reference_models() -> dict[str, object]:
    """Unfitted reference classifiers. Names are used in every table.

    The first three are the Task 3 models (same definitions as ``cortexhr.baselines.make_models``); the Random
    Forest reproduces the official benchmark row 'Random Forest (300 trees, balanced)'.
    """
    return {
        "Dummy (always stays)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (unweighted)": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
        "Logistic Regression (balanced) [Task 3 baseline]": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=config.RANDOM_STATE),
        "Decision Tree (depth 4, balanced)": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=config.RANDOM_STATE),
        "Random Forest (300 trees, balanced)": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=1),
    }


def oof_proba(estimator, X: pd.DataFrame, y: pd.Series, folds) -> np.ndarray:
    """P(left) for every row from a model that never saw that row; preprocessing is re-fitted per fold."""
    return cross_val_predict(estimator, X, y, cv=folds, method="predict_proba")[:, 1]


def fit_all_reference(X, y, folds) -> dict[str, np.ndarray]:
    return {name: oof_proba(model_pipeline(clf), X, y, folds) for name, clf in reference_models().items()}


def feature_importance(fitted_pipeline, top: int = 15) -> pd.DataFrame:
    """Gain importance of a fitted V2 pipeline, by transformed column name (used as a leakage sanity check)."""
    from cortexhr.features import output_feature_names
    names = output_feature_names(fitted_pipeline.named_steps["features"])
    gain = fitted_pipeline.named_steps["model"].booster_.feature_importance(importance_type="gain")
    df = pd.DataFrame({"feature": names, "gain": gain}).sort_values("gain", ascending=False)
    df["gain_share"] = df["gain"] / df["gain"].sum()
    return df.head(top).reset_index(drop=True)
