"""Tests for src/cortexhr/features.py  (Day 2, Task 2).

The three-row tests use SYNTHETIC records that follow the real attrition_dataset schema (all 42
columns plus the monthly features). They prove the code works; they do NOT prove anything about the
real data. The class ``TestRealDataset`` runs the same checks on the actual workbook and is skipped
when ``data/raw/02_CortexHR_AI_ML_Dataset.xlsx`` is not present.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from cortexhr.audit import LEAKAGE_COLUMNS, NON_FEATURE_COLUMNS, TARGET
from cortexhr.data_dictionary import D
from cortexhr.features import (CATEGORICAL_FEATURES, FORBIDDEN_COLUMNS, NUMERIC_FEATURES, UNCONFIRMED_FEATURES,
                               ForbiddenColumnError, RawCleaner, build_pipeline, feature_columns,
                               output_feature_names, prepare_training_rows, select_features)
from cortexhr.paths import WORKBOOK

MONTHLY = ["attendance_rate_change_3m", "avg_unpaid_absence_days_3m", "avg_variable_pay_3m"]


def _row(**over) -> dict:
    """One synthetic raw employee with the full real schema; keyword arguments override values."""
    base = {
        # identifiers / provenance
        "row_id": 1, "employee_id": "EMP-9001", "observation_date": pd.Timestamp("2025-09-30"), "source": "master_employee_snapshot",
        # categorical
        "department_id": "D01", "role_id": "R001", "work_location": "Bengaluru", "work_mode": "Hybrid",
        "employment_type": "Permanent", "highest_education": "B.Tech",
        # numeric
        "job_level": 2, "tenure_months": 30, "total_experience_years": 5.0, "previous_employers_count": 1,
        "commute_distance_km": 12.0, "annual_fixed_pay": 700000.0, "comp_ratio": 1.0, "raises_received": 2,
        "last_raise_pct": 8.0, "months_since_last_raise": 10, "promotions_received": 1, "months_since_last_promotion": 14.0,
        "internal_transfers": 0, "last_performance_rating": 3.0, "last_engagement_score": 3.5, "manager_changes_12m": 0,
        "training_hours_12m": 20.0, "attendance_rate_3m": 0.95, "late_arrivals_3m": 2, "avg_overtime_hours_per_month_3m": 4.0,
        "avg_hours_worked_3m": 8.4, "wfh_share_3m": 0.3, "unpaid_leave_days_6m": 0.0, "paid_leave_days_6m": 6,
        "dept_attrition_rate_prior_12m": 0.14,
        # monthly features (as produced by monthly.add_monthly_features)
        "attendance_rate_change_3m": 0.01, "avg_unpaid_absence_days_3m": 0.0, "avg_variable_pay_3m": 5000.0,
        # target and leaks: present in a RAW record, must never reach the model
        TARGET: 0, "voluntary_left_within_12m": 0, "exit_date": pd.NaT, "resignation_date": pd.NaT,
        "exit_reason": None, "exit_interview_done": None, "fnf_settlement_amount": np.nan,
    }
    base.update(over)
    return base


@pytest.fixture
def three_rows() -> pd.DataFrame:
    """SYNTHETIC: a normal row, a row full of blanks, and a messy row."""
    a = _row()
    b = _row(row_id=2, employee_id="EMP-9002", department_id="D02", role_id="R002", work_mode="Remote",
             highest_education=None,                                   # blank category
             commute_distance_km=np.nan, last_performance_rating=np.nan, last_engagement_score=np.nan,
             months_since_last_promotion=np.nan, promotions_received=0, last_raise_pct=np.nan, raises_received=0,
             attendance_rate_change_3m=np.nan,                         # only one month of data
             tenure_months=8, total_experience_years=0.5, annual_fixed_pay=400000.0,
             **{TARGET: 1, "voluntary_left_within_12m": 1, "exit_date": pd.Timestamp("2026-03-01"),
                "exit_reason": "Better opportunity", "exit_interview_done": "Y", "fnf_settlement_amount": 52000.0})
    c = _row(row_id=3, employee_id="EMP-9003", department_id="D01", role_id="R003", work_location="BENGALURU",
             attendance_rate_3m=1.12, wfh_share_3m=-0.02, tenure_months=60, total_experience_years=3.0,
             job_level=3, annual_fixed_pay=1200000.0, avg_variable_pay_3m=np.nan)   # experience < tenure, bad share
    return pd.DataFrame([a, b, c])


# ---------------------------------------------------------------- the three-row test (synthetic)
class TestThreeRows:
    def test_three_row_end_to_end(self, three_rows, capsys):
        raw = three_rows
        y = raw[TARGET]                                   # target stays outside the features
        X = select_features(raw)
        assert TARGET not in X.columns and len(X) == 3

        pipe = build_pipeline()
        out = pipe.fit_transform(X)
        names = output_feature_names(pipe)

        assert out.shape == (3, len(names))               # exactly three rows
        assert np.issubdtype(out.dtype, np.number)        # numerical output
        assert np.isfinite(out).all()                     # no NaN / inf left
        assert y.tolist() == [0, 1, 0]                    # target untouched by the pipeline
        # visible evidence for the report (run with -s to see it)
        print("\nTHREE-ROW TEST (synthetic rows, real schema)")
        print(f"  raw columns in : {raw.shape[1]}   input columns after select_features: {X.shape[1]}")
        print(f"  transformed    : shape {out.shape}, dtype {out.dtype}, NaN {int(np.isnan(out).sum())}, inf {int(np.isinf(out).sum())}")
        print(f"  output columns : {len(names)} (numeric/scaled, missing-indicators, one-hot, flags)")
        print(f"  target y       : {y.tolist()} (never passed to the pipeline)")

    def test_target_ids_and_leaks_are_not_inputs(self, three_rows):
        X = select_features(three_rows)
        assert not (set(X.columns) & FORBIDDEN_COLUMNS)
        pipe = build_pipeline().fit(X)
        names = " ".join(output_feature_names(pipe))
        for bad in [TARGET, "voluntary", "exit_", "resignation", "fnf_", "employee_id", "row_id", "observation_date", "source"]:
            assert bad not in names, f"'{bad}' appears in the output feature names"

    def test_missing_values_are_handled(self, three_rows):
        X = select_features(three_rows)
        assert X.isna().sum().sum() > 0                   # the input really has blanks
        pipe = build_pipeline().fit(X)
        out = pipe.transform(X)
        assert np.isfinite(out).all()
        names = output_feature_names(pipe)
        assert "cat__highest_education_Unknown" in names   # blank category -> its own "Unknown" level
        assert any("missingindicator_commute_distance_km" in n for n in names)  # "was missing" flag
        assert "flags__never_promoted" in names and "flags__never_raised" in names

    def test_scaling_and_one_hot_encoding(self, three_rows):
        pipe = build_pipeline().fit(select_features(three_rows))
        out = pipe.transform(select_features(three_rows))
        df = pd.DataFrame(out, columns=output_feature_names(pipe))
        tenure = df["num__tenure_months"]                  # scaled on the 3 fitted rows
        assert abs(tenure.mean()) < 1e-9 and abs(tenure.std(ddof=0) - 1) < 1e-9
        for col in CATEGORICAL_FEATURES:                   # exactly one 1 per categorical group per row
            group = df[[c for c in df.columns if c.startswith(f"cat__{col}_")]]
            assert set(np.unique(group.values)) <= {0.0, 1.0}
            assert (group.sum(axis=1) == 1).all()

    def test_unseen_categories_do_not_fail(self, three_rows):
        pipe = build_pipeline().fit(select_features(three_rows))
        new = pd.DataFrame([_row(employee_id="EMP-9100", department_id="D99", role_id="R999", work_mode="Mars")])
        out = pipe.transform(select_features(new))
        names = output_feature_names(pipe)
        assert out.shape == (1, len(names)) and np.isfinite(out).all()
        role_cols = [i for i, n in enumerate(names) if n.startswith("cat__role_id_")]
        assert out[0, role_cols].sum() == 0                # unseen role -> all zeros, no error

    def test_transform_new_records_after_fit(self, three_rows):
        pipe = build_pipeline().fit(select_features(three_rows))
        new = pd.DataFrame([_row(employee_id="EMP-9101"), _row(employee_id="EMP-9102", tenure_months=99)])
        out = pipe.transform(select_features(new))
        assert out.shape == (2, len(output_feature_names(pipe)))

    def test_raw_record_with_target_is_refused_by_pipeline(self, three_rows):
        with pytest.raises(ForbiddenColumnError, match=TARGET):
            build_pipeline().fit(three_rows)               # the full raw frame still contains the target


# ---------------------------------------------------------------- robustness (added in the Day 1/2 audit)
class TestRobustness:
    def test_column_order_does_not_matter(self, three_rows):
        X = select_features(three_rows)
        pipe = build_pipeline().fit(X)
        np.testing.assert_allclose(pipe.transform(X), pipe.transform(X[X.columns[::-1]]))

    def test_extra_harmless_column_is_ignored(self, three_rows):
        X = select_features(three_rows)
        pipe = build_pipeline().fit(X)
        np.testing.assert_allclose(pipe.transform(X.assign(free_text_note="x")), pipe.transform(X))

    def test_input_frame_is_not_modified(self, three_rows):
        X = select_features(three_rows)
        before = X.copy()
        build_pipeline().fit(X).transform(X)
        pd.testing.assert_frame_equal(X, before)

    def test_empty_input_gives_clear_error(self, three_rows):
        pipe = build_pipeline().fit(select_features(three_rows))
        with pytest.raises(ValueError, match="no rows"):
            pipe.transform(select_features(three_rows).iloc[:0])

    def test_completely_blank_monthly_columns_at_fit_give_clear_error(self, three_rows):
        X = select_features(three_rows)
        X[MONTHLY] = np.nan
        with pytest.raises(ValueError, match="completely blank.*use_monthly=False"):
            build_pipeline().fit(X)
        build_pipeline(use_monthly=False).fit(X.drop(columns=MONTHLY))      # the documented alternative works

    def test_pipeline_can_be_cloned_and_saved(self, three_rows, tmp_path):
        import joblib
        from sklearn.base import clone
        X = select_features(three_rows)
        pipe = build_pipeline().fit(X)
        assert clone(pipe).get_params().keys() == pipe.get_params().keys()
        joblib.dump(pipe, tmp_path / "p.joblib")
        np.testing.assert_allclose(joblib.load(tmp_path / "p.joblib").transform(X), pipe.transform(X))

    def test_usable_inside_a_model_pipeline_and_target_stays_separate(self, three_rows):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        X, y = select_features(three_rows), three_rows[TARGET]
        model = Pipeline([("features", build_pipeline()), ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
        assert model.predict_proba(X).shape == (3, 2)        # SYNTHETIC mechanics check, not a performance claim


# ---------------------------------------------------------------- guards
class TestForbiddenColumns:
    @pytest.mark.parametrize("col", [TARGET] + LEAKAGE_COLUMNS + NON_FEATURE_COLUMNS)
    def test_selecting_forbidden_column_raises(self, three_rows, col):
        with pytest.raises(ForbiddenColumnError) as err:
            select_features(three_rows, columns=["tenure_months", col])
        assert col in str(err.value) and "Forbidden" in str(err.value)

    def test_custom_pipeline_lists_are_checked_too(self):
        with pytest.raises(ForbiddenColumnError, match="exit_date"):
            build_pipeline(numeric_features=["tenure_months", "exit_date"])

    def test_missing_input_column_gives_helpful_error(self, three_rows):
        with pytest.raises(KeyError, match="tenure_months"):
            select_features(three_rows.drop(columns=["tenure_months"]))

    def test_unconfirmed_feature_is_off_by_default(self, three_rows):
        assert UNCONFIRMED_FEATURES[0] not in select_features(three_rows).columns
        assert UNCONFIRMED_FEATURES[0] in select_features(three_rows, include_unconfirmed=True).columns

    def test_feature_lists_match_data_dictionary(self):
        """Every column the Task 1 dictionary calls FEATURE is accounted for (none forgotten)."""
        dictionary_features = {c for c, v in D.items() if v[1] == "FEATURE"}
        configured = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) | set(UNCONFIRMED_FEATURES)
        assert configured == dictionary_features

    def test_monthly_columns_follow_window(self):
        assert feature_columns(use_monthly=False)["monthly"] == []
        assert feature_columns()["monthly"] == MONTHLY


# ---------------------------------------------------------------- fit on training data only
class TestFitOnTrainOnly:
    def test_scaler_statistics_come_from_training_rows_only(self, three_rows):
        X = select_features(three_rows)
        train, test = X.iloc[:2], X.iloc[2:]
        pipe = build_pipeline().fit(train)
        num = pipe.named_steps["preprocess"].named_transformers_["num"]
        i = NUMERIC_FEATURES.index("tenure_months")
        assert num.named_steps["scale"].mean_[i] == pytest.approx(train["tenure_months"].mean())   # not the 3-row mean
        out = pd.DataFrame(pipe.transform(test), columns=output_feature_names(pipe))
        expected = (test["tenure_months"].iloc[0] - train["tenure_months"].mean()) / train["tenure_months"].std(ddof=0)
        assert out["num__tenure_months"].iloc[0] == pytest.approx(expected)

    def test_transform_never_changes_fitted_state(self, three_rows):
        X = select_features(three_rows)
        pipe = build_pipeline().fit(X.iloc[:2])
        before = copy.deepcopy(pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"].mean_)
        medians_before = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"].statistics_.copy()
        pipe.transform(X.iloc[2:])
        scale = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
        impute = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
        np.testing.assert_array_equal(scale.mean_, before)
        np.testing.assert_array_equal(impute.statistics_, medians_before)


# ---------------------------------------------------------------- cleaning rules (Task 1 decisions)
class TestRawCleaner:
    def _fit_clean(self, X, **kw):
        return RawCleaner(**kw).fit(X).transform(X)

    def test_task1_rules(self):
        rows = [_row(role_id="R001", annual_fixed_pay=500000.0), _row(role_id="R001", annual_fixed_pay=520000.0),
                _row(role_id="R001", annual_fixed_pay=5100000.0),                          # 10x typo
                _row(role_id="R001", work_location="Madras", attendance_rate_3m=1.12, wfh_share_3m=1.04,
                     tenure_months=60, total_experience_years=3.0, months_since_last_promotion=np.nan, last_raise_pct=np.nan)]
        X = select_features(pd.DataFrame(rows))
        out = self._fit_clean(X)
        assert out["annual_fixed_pay"].iloc[2] == pytest.approx(np.log1p(510000.0))       # typo fixed (÷10), then log1p
        assert out["work_location"].iloc[3] == "Chennai"                                   # D02
        assert out["attendance_rate_3m"].iloc[3] == 1.0 and out["wfh_share_3m"].iloc[3] == 1.0   # D04/D05
        assert out["total_experience_years"].iloc[3] == pytest.approx(5.0)                 # D06: floor at tenure 60m = 5y
        assert out["never_promoted"].tolist() == [0, 0, 0, 1] and out["months_since_last_promotion"].iloc[3] == 60
        assert out["never_raised"].tolist() == [0, 0, 0, 1] and out["last_raise_pct"].iloc[3] == 0

    def test_tenure_is_not_silently_corrected(self):
        X = select_features(pd.DataFrame([_row(tenure_months=-4), _row(tenure_months=10)]))
        assert RawCleaner().fit(X).transform(X)["tenure_months"].tolist() == [-4, 10]      # open item O1

    def test_non_numeric_text_in_numeric_column_is_reported(self):
        X = select_features(pd.DataFrame([_row(), _row()]))
        X["tenure_months"] = X["tenure_months"].astype(object)
        X.loc[1, "tenure_months"] = "abc"
        with pytest.raises(ValueError, match="tenure_months"):
            RawCleaner(numeric_columns=["tenure_months"]).fit(X).transform(X)

    def test_plain_arrays_are_rejected(self):
        with pytest.raises(TypeError, match="DataFrame"):
            build_pipeline().fit(np.zeros((3, 3)))


class TestPrepareTrainingRows:
    def test_duplicates_and_negative_tenure(self):
        rows = pd.DataFrame([_row(), _row(), _row(row_id=3, employee_id="EMP-9003", tenure_months=-2), _row(row_id=4, employee_id="EMP-9004")])
        rows.loc[1, "row_id"] = 99                       # same content, different row counter -> still a duplicate
        out, rep = prepare_training_rows(rows)
        assert rep["duplicates_dropped"] == 1 and rep["negative_tenure_excluded"] == 1
        assert rep["negative_tenure_ids"] == ["EMP-9003"] and len(out) == 2


# ---------------------------------------------------------------- the same checks on the REAL data
@pytest.fixture(scope="module")
def real():
    if not WORKBOOK.exists():
        pytest.skip("real workbook not present in data/raw/")
    from cortexhr.audit import load_attrition
    from cortexhr.monthly import add_monthly_features, load_monthly_tables
    raw = load_attrition()
    att, pay = load_monthly_tables()
    rows, report = prepare_training_rows(raw)
    rows = rows.reset_index(drop=True)
    table, diag = add_monthly_features(rows, att, pay)
    return raw, rows, table, diag, report


@pytest.mark.skipif(not WORKBOOK.exists(), reason="real workbook not present in data/raw/")
class TestRealDataset:

    def test_schema_and_row_counts(self, real):
        raw, rows, table, diag, report = real
        assert raw.shape == (2389, 42)
        assert len(table) == len(rows) == 2379            # join did not multiply or lose rows
        assert report["duplicates_dropped"] == 6 and report["negative_tenure_excluded"] == 4

    def test_three_real_rows_through_a_pipeline_fitted_on_train(self, real):
        from sklearn.model_selection import train_test_split
        _, _, table, _, _ = real
        X, y = select_features(table), table[TARGET]
        assert not (set(X.columns) & FORBIDDEN_COLUMNS) and TARGET not in X.columns
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        pipe = build_pipeline().fit(X_train)
        three = table.iloc[[0, 1, 2000]]                  # two real snapshot rows + one simulated HIST row (no monthly data)
        out = pipe.transform(select_features(three))
        assert out.shape == (3, len(output_feature_names(pipe)))
        assert np.isfinite(out).all()
        full = pipe.transform(X_test)
        assert full.shape[0] == len(X_test) and np.isfinite(full).all()

    def test_model_pipeline_runs_and_test_rows_are_only_transformed(self, real):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        _, _, table, _, _ = real
        X, y = select_features(table), table[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model = Pipeline([("features", build_pipeline()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        model.fit(X_train, y_train)
        assert model.predict_proba(X_test).shape == (len(X_test), 2)
        # the scaler was fitted on the training rows only
        num = model.named_steps["features"].named_steps["preprocess"].named_transformers_["num"]
        assert num.named_steps["scale"].n_samples_seen_ == len(X_train)

    def test_monthly_window_has_no_look_ahead(self, real):
        _, _, table, diag, _ = real
        assert diag["no_look_ahead"] is True
        assert diag["latest_month_used"] == "2025-09-01"  # no month after the 30 Sep 2025 snapshot
        assert diag["duplicate_rows_removed"]["payroll_monthly_features"] == 15
        assert diag["rows_with_attendance_data"] == 379 and diag["rows_without_attendance_data"] == 2000
