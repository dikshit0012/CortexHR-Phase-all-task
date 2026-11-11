"""Task 2 stand-in: the reusable feature pipeline.

If you already have your real Task 2 pipeline, keep it and make
`build_feature_pipeline()` return it (or import yours here). Task 3 only needs:
  * TARGET
  * select_features(df) -> X, y
  * make_model(estimator) -> Pipeline(features -> estimator)
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "left_within_12m"

# Columns that must NEVER be model inputs, with the reason (shown in the notebook).
EXCLUDED_COLUMNS = {
    "employee_id": "identifier: lets a model memorise individuals, carries no real signal",
    "exit_interview_completed": "post-outcome: an exit interview only happens after someone leaves",
}

NUMERIC_COLUMNS = [
    "age", "job_level", "monthly_income", "years_at_company",
    "years_since_last_promotion", "distance_from_home_km", "job_satisfaction",
    "work_life_balance", "num_companies_worked", "performance_rating",
    "training_hours_last_year",
]
CATEGORICAL_COLUMNS = ["department", "overtime"]


def build_feature_pipeline() -> ColumnTransformer:
    """Impute, scale and one-hot encode. Unfitted: it is fitted inside each CV fold."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_COLUMNS),
                              ("cat", categorical, CATEGORICAL_COLUMNS)])


def select_features(df: pd.DataFrame, extra_columns: list[str] | None = None):
    """Return raw X (approved columns only) and y. `extra_columns` is for leakage demos."""
    if TARGET not in df.columns:
        raise KeyError(f"target column {TARGET!r} not found")
    columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + list(extra_columns or [])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"expected columns missing from the data: {missing}")

    # Explicitly handle missing target values
    valid_mask = df[TARGET].notna()
    if not valid_mask.all():
        n_dropped = int((~valid_mask).sum())
        import warnings
        warnings.warn(f"Dropped {n_dropped} row(s) with missing target {TARGET!r}.", UserWarning)

    df_valid = df.loc[valid_mask]
    y_raw = df_valid[TARGET]
    unique_vals = set(y_raw.unique())
    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Target {TARGET!r} must contain valid binary {0, 1} labels; found: {unique_vals}")

    X = df_valid[columns].copy()
    y = y_raw.astype(int)
    assert TARGET not in X.columns, "target leaked into X"
    return X, y



def make_model(estimator, extra_categorical: list[str] | None = None) -> Pipeline:
    """Feature pipeline + estimator in ONE Pipeline, so CV re-fits preprocessing per fold."""
    features = build_feature_pipeline()
    if extra_categorical:  # only used by the deliberate leakage demonstration
        features.transformers.append(
            ("extra", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             extra_categorical))
    return Pipeline([("features", features), ("model", estimator)])
