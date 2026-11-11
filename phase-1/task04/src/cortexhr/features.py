"""Reusable feature pipeline for the CortexHR attrition project (Day 2, Task 2).

    raw rows ──► select_features() ──► build_pipeline() ──► model-ready numbers

* ``select_features``  picks the allowed columns and refuses the target, identifiers and leaks.
* ``RawCleaner``       applies the Task 1 cleaning rules inside the pipeline, so training and
                       future scoring use exactly the same code.
* ``build_pipeline``   returns a scikit-learn ``Pipeline`` = RawCleaner + ColumnTransformer.

Fit on TRAINING data only, then reuse the fitted object for test data and new employees::

    pipe = build_pipeline()
    pipe.fit(X_train)
    X_train_p = pipe.transform(X_train)
    X_test_p = pipe.transform(X_test)

The target is never an input; it is kept in a separate Series ``y``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .audit import LEAKAGE_COLUMNS, NON_FEATURE_COLUMNS, TARGET, normalise_location
from .monthly import WINDOW_MONTHS, monthly_feature_names

# ----------------------------------------------------------------------------------------------
# 1. Which columns may enter the model (decided in the Task 1 audit, see docs/decisions.md)
# ----------------------------------------------------------------------------------------------
# Never allowed as an input, with the reason shown in the error message.
FORBIDDEN_REASONS: dict[str, str] = {TARGET: "it is the target (the answer we are predicting)"}
FORBIDDEN_REASONS.update({c: "it leaks the outcome: it only exists after the person has left (Task 1, D18/D19)"
                          for c in LEAKAGE_COLUMNS})
FORBIDDEN_REASONS.update({
    "row_id": "it is a row counter, not an employee trait",
    "employee_id": "it is an identifier (use it only to group or join)",
    "source": "it records which simulation produced the row, not anything about the employee",
    "observation_date": "it is the prediction date; use it for time windows and splits, not as a feature",
})
FORBIDDEN_COLUMNS = frozenset(FORBIDDEN_REASONS)
assert set(LEAKAGE_COLUMNS) | set(NON_FEATURE_COLUMNS) | {TARGET} == set(FORBIDDEN_COLUMNS)

# Documented as backward-looking but NOT confirmed by the Python team (open item O2 / decision D24).
UNCONFIRMED_FEATURES = ["dept_attrition_rate_prior_12m"]

CATEGORICAL_FEATURES = ["department_id", "role_id", "work_location", "work_mode",
                        "employment_type", "highest_education"]

NUMERIC_FEATURES = [
    "job_level", "tenure_months", "total_experience_years", "previous_employers_count",
    "commute_distance_km", "annual_fixed_pay", "comp_ratio", "raises_received", "last_raise_pct",
    "months_since_last_raise", "promotions_received", "months_since_last_promotion", "internal_transfers",
    "last_performance_rating", "last_engagement_score", "manager_changes_12m", "training_hours_12m",
    "attendance_rate_3m", "late_arrivals_3m", "avg_overtime_hours_per_month_3m", "avg_hours_worked_3m",
    "wfh_share_3m", "unpaid_leave_days_6m", "paid_leave_days_6m",
]

# Heavy right tail (Task 1, D08/D10): compressed with log1p before scaling.
LOG_PREFIXES = ("annual_fixed_pay", "avg_overtime_hours_per_month_3m", "avg_variable_pay_")


class ForbiddenColumnError(ValueError):
    """Raised when the target, an identifier or a leaking column is used as a model input."""


def check_no_forbidden(columns: Sequence[str], context: str = "feature set") -> None:
    """Raise a helpful error if any forbidden column is in ``columns``."""
    bad = [c for c in columns if c in FORBIDDEN_COLUMNS]
    if bad:
        lines = "\n".join(f"  - {c}: {FORBIDDEN_REASONS[c]}" for c in bad)
        raise ForbiddenColumnError(
            f"Forbidden column(s) in the {context}:\n{lines}\n"
            "Remove them from the inputs (keep the target in a separate variable y).")


def feature_columns(include_unconfirmed: bool = False, use_monthly: bool = True,
                    window_months: int = WINDOW_MONTHS) -> dict[str, list[str]]:
    """The model-input columns, by group: ``numeric``, ``monthly``, ``categorical``."""
    numeric = list(NUMERIC_FEATURES) + (list(UNCONFIRMED_FEATURES) if include_unconfirmed else [])
    monthly = monthly_feature_names(window_months)["model"] if use_monthly else []
    cols = {"numeric": numeric, "monthly": monthly, "categorical": list(CATEGORICAL_FEATURES)}
    check_no_forbidden(sum(cols.values(), []), "configured feature lists")
    return cols


def select_features(df: pd.DataFrame, columns: Sequence[str] | None = None,
                    include_unconfirmed: bool = False, use_monthly: bool = True,
                    window_months: int = WINDOW_MONTHS) -> pd.DataFrame:
    """Return the allowed input columns of ``df`` (a copy). The target is NOT included.

    ``columns`` overrides the default list but is still checked against the forbidden set, so
    ``select_features(df, columns=["tenure_months", "left_within_12m"])`` raises.
    """
    if columns is None:
        groups = feature_columns(include_unconfirmed, use_monthly, window_months)
        columns = sum(groups.values(), [])
    columns = list(columns)
    check_no_forbidden(columns, "requested feature selection")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        hint = (" (monthly features come from monthly.add_monthly_features)"
                if any(c.startswith(("attendance_rate_change", "avg_")) for c in missing) else "")
        raise KeyError(f"Columns missing from the data: {missing}{hint}")
    return df[columns].copy()


# ----------------------------------------------------------------------------------------------
# 2. Row filtering from Task 1 (rows are removed, so this cannot live inside a sklearn Pipeline)
# ----------------------------------------------------------------------------------------------
def prepare_training_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the row-level Task 1 decisions before splitting/training.

    * D01: drop exact duplicate rows (ignoring ``row_id``).
    * D03 / open item O1: exclude rows with negative tenure until the Python team confirms the
      true join dates. They are NOT corrected or imputed.
    """
    content = [c for c in df.columns if c != "row_id"]
    n0 = len(df)
    out = df.drop_duplicates(subset=content, keep="first")
    n_dup = n0 - len(out)
    bad_tenure = out["tenure_months"] < 0
    report = {"rows_in": n0, "duplicates_dropped": int(n_dup),
              "negative_tenure_excluded": int(bad_tenure.sum()),
              "negative_tenure_ids": out.loc[bad_tenure, "employee_id"].tolist() if "employee_id" in out else []}
    out = out.loc[~bad_tenure].copy()
    report["rows_out"] = len(out)
    return out, report


# ----------------------------------------------------------------------------------------------
# 3. Cleaning rules inside the pipeline
# ----------------------------------------------------------------------------------------------
class RawCleaner(BaseEstimator, TransformerMixin):
    """Applies the Task 1 cleaning rules. The only thing it learns in ``fit`` is the median pay
    mid-point per role (used to spot 10x salary typos), so it is fitted on training data only.

    Rules (each applies only when its columns are present):
      D02  one spelling per city               D06  total experience >= tenure (in years)
      D04  attendance_rate_3m clipped to 0-1   D07  pay / comp_ratio > 3x the role median -> pay / 10
      D05  wfh_share_3m clipped to 0-1         D08/D10  log1p on pay and overtime columns
      D11  blank promotion months = never promoted -> filled with tenure, flag ``never_promoted``
      D12  blank raise % = never had a raise -> filled with 0, flag ``never_raised``
    Tenure itself is NOT altered (open item O1).
    """

    def __init__(self, required_columns: Sequence[str] = (), numeric_columns: Sequence[str] = (),
                 categorical_columns: Sequence[str] = (), non_empty_columns: Sequence[str] = ()):
        self.required_columns = required_columns
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.non_empty_columns = non_empty_columns     # must have >= 1 value in the data used for fit()

    # -- helpers ------------------------------------------------------------------------------
    def _check(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The pipeline expects a pandas DataFrame with named columns, "
                            f"got {type(X).__name__}.")
        check_no_forbidden(list(X.columns), "pipeline input")
        if len(X) == 0:
            raise ValueError("Pipeline input has no rows. Pass at least one employee record.")
        missing = [c for c in self.required_columns if c not in X.columns]
        if missing:
            raise KeyError(f"Pipeline input is missing required column(s): {missing}")
        return X

    # -- sklearn API --------------------------------------------------------------------------
    def fit(self, X, y=None):
        X = self._check(X)
        empty = [c for c in self.non_empty_columns if c in X.columns and pd.to_numeric(X[c], errors="coerce").isna().all()]
        if empty:
            raise ValueError(
                f"These numeric column(s) are completely blank in the data used for fit(): {empty}. A median cannot be "
                "learned from nothing (and silently dropping them would change the output columns). If they are the "
                "monthly attendance/payroll features, build the table with monthly.add_monthly_features first or use "
                "build_pipeline(use_monthly=False).")
        self.role_mid_median_ = pd.Series(dtype=float)
        if {"annual_fixed_pay", "comp_ratio", "role_id"} <= set(X.columns):
            mid = pd.to_numeric(X["annual_fixed_pay"]) / pd.to_numeric(X["comp_ratio"]).replace(0, np.nan)
            self.role_mid_median_ = mid.groupby(X["role_id"]).median()
        return self

    def transform(self, X):
        X = self._check(X).copy()
        for c in self.numeric_columns:
            if c in X.columns:
                try:
                    X[c] = pd.to_numeric(X[c], errors="raise")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Column '{c}' must be numeric: {e}") from e
        for c in self.categorical_columns:           # None / pd.NA -> np.nan so the imputer sees them
            if c in X.columns:
                X[c] = X[c].astype(object).where(X[c].notna(), np.nan)

        if "work_location" in X:
            X["work_location"] = X["work_location"].map(normalise_location)
        for c in ("attendance_rate_3m", "wfh_share_3m"):
            if c in X:
                X[c] = X[c].clip(0, 1)
        if {"total_experience_years", "tenure_months"} <= set(X.columns):
            floor = X["tenure_months"].clip(lower=0) / 12
            X["total_experience_years"] = np.maximum(X["total_experience_years"], floor)
        if {"annual_fixed_pay", "comp_ratio", "role_id"} <= set(X.columns) and len(self.role_mid_median_):
            ratio = (X["annual_fixed_pay"] / X["comp_ratio"].replace(0, np.nan)) / X["role_id"].map(self.role_mid_median_)
            X["annual_fixed_pay"] = np.where(ratio > 3, X["annual_fixed_pay"] / 10, X["annual_fixed_pay"])
        if "months_since_last_promotion" in X:
            X["never_promoted"] = X["months_since_last_promotion"].isna().astype(int)
            if "tenure_months" in X:
                X["months_since_last_promotion"] = X["months_since_last_promotion"].fillna(X["tenure_months"].clip(lower=0))
        if "last_raise_pct" in X:
            X["never_raised"] = X["last_raise_pct"].isna().astype(int)
            X["last_raise_pct"] = X["last_raise_pct"].fillna(0)
        for c in X.columns:
            if c.startswith(LOG_PREFIXES):
                X[c] = np.log1p(X[c].clip(lower=0))
        return X


# ----------------------------------------------------------------------------------------------
# 4. The pipeline
# ----------------------------------------------------------------------------------------------
def build_pipeline(include_unconfirmed: bool = False, use_monthly: bool = True,
                   window_months: int = WINDOW_MONTHS,
                   numeric_features: Sequence[str] | None = None,
                   categorical_features: Sequence[str] | None = None) -> Pipeline:
    """Build the (unfitted) feature pipeline.

    Parameters
    ----------
    include_unconfirmed : add ``dept_attrition_rate_prior_12m``. Default False because the Python
        team has not confirmed it only uses exits before the observation date (open item O2).
        Run models with and without it until that is answered.
    use_monthly : include the point-in-time attendance/payroll features from ``monthly.py``.
    window_months : length of the monthly window (names of the monthly columns follow it).
    numeric_features, categorical_features : optional overrides of the default lists. They are
        still checked against the forbidden columns.

    Returns
    -------
    sklearn Pipeline ``[("clean", RawCleaner), ("preprocess", ColumnTransformer)]``. Call
    ``.fit(X_train)`` once, then ``.transform(...)`` for any later data. Output is a dense float
    array; column names via :func:`output_feature_names`.
    """
    groups = feature_columns(include_unconfirmed, use_monthly, window_months)
    numeric = list(numeric_features) if numeric_features is not None else groups["numeric"]
    categorical = list(categorical_features) if categorical_features is not None else groups["categorical"]
    monthly = groups["monthly"]
    check_no_forbidden(numeric + categorical + monthly, "pipeline configuration")

    flags = [f for f, src in (("never_promoted", "months_since_last_promotion"), ("never_raised", "last_raise_pct"))
             if src in numeric]

    numeric_steps = Pipeline([
        # median is robust to the long tails seen in Task 1; add_indicator appends a 0/1 "was missing"
        # column for every numeric column that had blanks in the TRAINING data
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    # No "was missing" flag here: these features are blank for ALL simulated HIST rows. NOTE this does not hide
    # the data source, because the constant median fill identifies those rows just as well (see the README
    # limitations); the flag is omitted only to avoid adding a redundant column.
    monthly_steps = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_steps = Pipeline([
        # blanks (e.g. highest_education) become their own "Unknown" category instead of guessing the mode
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    blocks = [("num", numeric_steps, numeric)]
    if monthly:
        blocks.append(("monthly", monthly_steps, monthly))
    blocks.append(("cat", categorical_steps, categorical))
    if flags:
        blocks.append(("flags", "passthrough", flags))
    preprocess = ColumnTransformer(blocks, remainder="drop", sparse_threshold=0)

    cleaner = RawCleaner(required_columns=numeric + monthly + categorical,
                         numeric_columns=numeric + monthly, categorical_columns=categorical,
                         non_empty_columns=numeric + monthly)
    return Pipeline([("clean", cleaner), ("preprocess", preprocess)])


def output_feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the columns produced by a FITTED pipeline."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
