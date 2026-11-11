"""Preprocessing, the three models, the split and training-only cross-validation. Preprocessing lives INSIDE each pipeline, so it is fitted on
training rows only and re-used unchanged for any later data (no train/test inconsistency)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

BASELINE, LINEAR, BOOSTING = "average_by_role_baseline", "linear_regression", "gradient_boosting"
NAMES = {BASELINE: "Average-by-role baseline", LINEAR: "Linear Regression", BOOSTING: "Gradient Boosting Regression"}


class RoleMeanBaseline(BaseEstimator, RegressorMixin):
    """Predicts the TRAINING average salary of the row's role; unseen roles get the overall TRAINING average (documented fallback)."""

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y, float), index=X.index)
        self.role_means_ = y.groupby(X["role_id"]).mean().to_dict()
        self.overall_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame):
        return X["role_id"].map(self.role_means_).fillna(self.overall_mean_).to_numpy(float)


def make_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    transformers = [("num", num, numeric)] if numeric else []
    transformers += [("cat", cat, categorical)] if categorical else []
    return ColumnTransformer(transformers, remainder="drop")


def build_model(name: str, numeric: List[str] = config.NUMERIC, categorical: List[str] = config.CATEGORICAL):
    if name == BASELINE:
        return RoleMeanBaseline()
    est = LinearRegression() if name == LINEAR else GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=config.SEED)
    if name not in (LINEAR, BOOSTING):
        raise ValueError(name)
    return Pipeline([("prep", make_preprocessor(numeric, categorical)), ("model", est)])


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Random 80/20 split, stratified by data source (real vs extension rows). hire_id is unique, so no person appears on both sides."""
    tr, te = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=df["source"] if "source" in df else None)
    return tr.sort_index(), te.sort_index()


def metrics(y, pred) -> Dict[str, float]:
    return {"mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}


def cv_oof(name: str, X: pd.DataFrame, y, **kw) -> np.ndarray:
    """Out-of-fold predictions on the TRAINING rows only (5-fold). Used for model selection and for the salary-band residuals."""
    cv = KFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    return cross_val_predict(build_model(name, **kw), X, y, cv=cv)


def feature_names(pipe: Pipeline) -> List[str]:
    return list(pipe.named_steps["prep"].get_feature_names_out())
