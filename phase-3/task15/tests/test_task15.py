"""Task 15 tests (general behaviour and properties; no metric values are hard-coded)."""
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cortexhr_task15 import config
from cortexhr_task15.bands import coverage, fit_calibration, fit_level_calibration, make_bands, make_bands_by_level
from cortexhr_task15.cortexhr_helpers import group_gap as official_group_gap
from cortexhr_task15.data import DataError, audit, detect_target, validate
from cortexhr_task15.fairness import (UnlockMissing, attach_gender, career_gap_groups, group_gap_checked, group_table, load_unlock, proxy_check)
from cortexhr_task15.models import BASELINE, BOOSTING, LINEAR, RoleMeanBaseline, build_model, make_preprocessor, metrics, split


# ------------------------------------------------------------------ dataset
def test_target_is_detected_from_the_dictionary_and_from_names(df):
    assert detect_target(df, {"offered_annual_fixed": "TARGET: annual fixed pay offered at hire (INR).", "band_mid_2026": "Role band mid-point"}) == config.TARGET
    assert detect_target(df) == config.TARGET
    with pytest.raises(DataError):
        detect_target(pd.DataFrame({"a": [1.0], "b": [2.0]}))


def test_validation_catches_bad_salary_data(df):
    assert validate(df)["rows"] == len(df)
    with pytest.raises(DataError):
        validate(df.assign(**{config.TARGET: df[config.TARGET].where(df.index != 0, -5)}))
    with pytest.raises(DataError):
        validate(df.assign(**{config.TARGET: df[config.TARGET].astype(float).where(df.index != 0, np.nan)}))
    with pytest.raises(DataError):
        validate(pd.concat([df, df.iloc[[0]]], ignore_index=True))
    with pytest.raises(DataError):
        validate(df.drop(columns=["role_id"]))


def test_audit_reports_real_structure(df):
    a = audit(df)
    assert a["rows"] == len(df) and a["duplicate_rows"] == 0 and a["duplicate_ids"] == 0 and a["non_positive"] == 0
    assert a["gender_in_table"] is False and a["historical_salary_field"] == []
    assert a["employment_type_present"] is False and a["employment_type_distribution"] == {}
    assert sum(a["role_distribution"].values()) == len(df)
    assert sum(a["department_distribution"].values()) == len(df)
    assert a["band_circularity"]["log_correlation"] > 0.95                      # why the 2026 bands are excluded


def test_every_column_has_a_feature_decision_and_no_sensitive_feature(df):
    assert set(df.columns) <= set(config.POLICY)
    assert not (set(config.ALLOWED) & (config.SENSITIVE_NEVER_FEATURES | {"band_min_2026", "band_mid_2026", "band_max_2026", config.TARGET, "hire_id", "source"}))
    assert config.POLICY["career_gap_months"][0] == "audit-only" and "career_gap_months" not in config.ALLOWED
    assert all(config.POLICY[c][0] == "audit-only" for c in ("gender", "age_years", "age_band", "person_ref", "record_type"))


# ------------------------------------------------------------------ preprocessing
def test_preprocessing_handles_missing_and_unseen_values(df):
    tr, _ = split(df)
    X = tr[config.ALLOWED].copy()
    prep = make_preprocessor(config.NUMERIC, config.CATEGORICAL).fit(X)
    Z = X.head(5).copy()
    Z.loc[Z.index[0], "prior_experience_years"] = np.nan
    Z.loc[Z.index[1], "highest_education"] = np.nan
    Z.loc[Z.index[2], "role_id"] = "R-NEW-NEVER-SEEN"
    out = prep.transform(Z)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert np.isfinite(out).all() and out.shape[0] == 5


def test_preprocessing_is_fitted_on_training_rows_only_and_is_reproducible(df):
    tr, te = split(df)
    p1 = build_model(LINEAR).fit(tr[config.ALLOWED], tr[config.TARGET])
    p2 = build_model(LINEAR).fit(tr[config.ALLOWED], tr[config.TARGET])
    np.testing.assert_allclose(p1.predict(te[config.ALLOWED]), p2.predict(te[config.ALLOWED]))
    scaler = p1.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_[0], tr["job_level"].mean())               # NOT the full-data or test mean
    np.testing.assert_allclose(scaler.mean_[1], tr["prior_experience_years"].mean())
    assert scaler.n_samples_seen_ == len(tr)


def test_split_is_reproducible_disjoint_and_sized(df):
    a, b = split(df), split(df)
    assert list(a[1].index) == list(b[1].index)
    assert not (set(a[0][config.ID_COL]) & set(a[1][config.ID_COL])) and len(a[0]) + len(a[1]) == len(df)
    assert abs(len(a[1]) / len(df) - config.TEST_SIZE) < 0.01


# ------------------------------------------------------------------ models
def test_baseline_uses_training_role_means_and_documented_fallback():
    X = pd.DataFrame({"role_id": ["A", "A", "B"]}); y = [100.0, 200.0, 400.0]
    m = RoleMeanBaseline().fit(X, y)
    np.testing.assert_allclose(m.predict(pd.DataFrame({"role_id": ["A", "B", "UNSEEN"]})), [150.0, 400.0, np.mean(y)])


def test_baseline_never_sees_test_salaries(df):
    tr, te = split(df)
    a = RoleMeanBaseline().fit(tr[config.ALLOWED], tr[config.TARGET]).predict(te[config.ALLOWED])
    te2 = te.assign(**{config.TARGET: te[config.TARGET] * 10})
    b = RoleMeanBaseline().fit(tr[config.ALLOWED], tr[config.TARGET]).predict(te2[config.ALLOWED])
    np.testing.assert_array_equal(a, b)


def test_three_models_predict_finite_salaries_on_the_held_out_set(result):
    te = result["test"]
    for m, p in result["pred"].items():
        assert len(p) == len(te) and np.isfinite(p).all(), m
    for m in (BASELINE, BOOSTING):                                              # an average of positive salaries can never be <= 0
        assert (result["pred"][m] > 0).all(), m
    assert (result["pred"][result["selected"]] > 0).all()                       # the model that produces the bands
    neg = result["summary"]["non_positive_test_predictions"]                    # Linear Regression can extrapolate below zero: reported, not hidden
    assert set(neg) == {BASELINE, LINEAR, BOOSTING} and neg[BASELINE] == 0 and neg[BOOSTING] == 0
    comp = result["comparison"].set_index("Model")
    for m in (BASELINE, LINEAR, BOOSTING):
        got = metrics(te[config.TARGET], result["pred"][m])
        name = {"average_by_role_baseline": "Average-by-role baseline", "linear_regression": "Linear Regression", "gradient_boosting": "Gradient Boosting Regression"}[m]
        assert comp.loc[name, "MAE"] == pytest.approx(got["mae"]) and comp.loc[name, "R2"] == pytest.approx(got["r2"])


def test_metrics_are_evaluated_on_test_rows_not_training_rows(result):
    c = result["comparison"]
    assert (c["MAE"] != c["train_MAE"]).all()                                   # test and train numbers differ
    assert result["selected"] in (LINEAR, BOOSTING) and bool(c.set_index("Model")["selected"].sum() == 1)


def test_model_selection_used_training_cross_validation_only(result):
    c = result["comparison"].set_index("Model")
    best = c.loc[["Linear Regression", "Gradient Boosting Regression"], "CV_MAE_train_only"].idxmin()
    assert c.loc[best, "selected"]


# ------------------------------------------------------------------ fairness
def test_career_gap_buckets():
    g = career_gap_groups(pd.Series([0, 3, 9, 11, 12, 36, np.nan]))
    assert list(g[:6]) == ["No career gap", "Career gap 3-9 months", "Career gap 3-9 months", "Career gap 3-9 months", "Career gap 12+ months", "Career gap 12+ months"] and pd.isna(g.iloc[6])


def test_official_group_gap_is_what_is_used():
    s, g = [1.0, 3.0, 10.0, 20.0], ["a", "a", "b", "b"]
    assert group_gap_checked(s, g)["max_gap"] == official_group_gap(s, g)[1] == 13.0
    import cortexhr_task15.fairness as F
    assert F.official_group_gap is official_group_gap
    assert Path(inspect.getsourcefile(official_group_gap)).resolve() == (
        config.ROOT.parent / "task12" / "src" / "cortexhr_task12" / "cortexhr_helpers.py"
    ).resolve()


def test_missing_group_labels_and_scores_are_dropped_and_counted():
    r = group_gap_checked([1.0, 2.0, 3.0, np.nan, 9.0], ["a", "a", None, "b", "b"])
    assert r["dropped_missing"] == 2 and r["group_means"] == {"a": 1.5, "b": 9.0}
    blank = group_gap_checked([1.0, 2.0, 3.0], ["a", " ", "b"])
    assert blank["dropped_missing"] == 1 and blank["group_means"] == {"a": 1.0, "b": 3.0}
    assert np.isnan(group_gap_checked([1.0, 2.0], ["a", "a"])["max_gap"])             # one group only: no gap, no crash
    assert np.isnan(group_gap_checked([1.0], [None])["max_gap"])


def test_small_groups_are_flagged_unreliable():
    f = pd.DataFrame({"g": ["big"] * config.MIN_GROUP + ["tiny"] * 3, "actual": 100.0, "predicted": 110.0, "band_ref": 100.0})
    t = group_table(f, "g", "m").set_index("group")
    assert bool(t.loc["big", "reliable_n"]) and not bool(t.loc["tiny", "reliable_n"]) and t.loc["tiny", "n"] == 3


def test_career_gap_tables_are_complete_and_gender_is_reported_when_unavailable(result):
    t = result["group_tables"]["career_gap"]
    assert set(t["group"]) == {"No career gap", "Career gap 3-9 months", "Career gap 12+ months"} and (t.groupby("model")["n"].sum() == len(result["test"])).all()
    s = result["gap_summaries"]["career_gap"]
    assert {"actual salary (historical pay)", "predicted salary (model)", "signed error (predicted - actual)", "absolute error", "recommended mid"} <= set(s["quantity"])
    if not result["summary"]["gender_audit"]["run"]:
        assert (result["dir"] / "fairness_gender_NOT_RUN.md").exists() and "gender" not in result["group_tables"]


def test_missing_unlock_file_is_reported_not_invented(tmp_path):
    assert load_unlock(tmp_path / "nope.csv") is None
    with pytest.raises(UnlockMissing):
        load_unlock(tmp_path / "nope.csv", required=True)


def test_unlock_schema_is_validated_and_unmatched_hires_stay_unlabelled(tmp_path, df):
    bad = tmp_path / "u.csv"; pd.DataFrame({"person_ref": ["a"], "age": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_unlock(bad)
    good = tmp_path / "g.csv"; pd.DataFrame({"person_ref": [df[config.ID_COL].iloc[0]], "record_type": ["x"], "gender": ["Female"]}).to_csv(good, index=False)
    out = attach_gender(df.head(3), load_unlock(good))
    assert out["gender"].tolist()[0] == "Female" and out["gender"].isna().sum() == 2
    duplicate = tmp_path / "duplicate.csv"
    pd.DataFrame({"person_ref": ["x", "x"], "gender": ["Female", "Male"]}).to_csv(duplicate, index=False)
    with pytest.raises(ValueError, match="duplicate person_ref"):
        load_unlock(duplicate)


def test_proxy_check_returns_auc_and_handles_tiny_groups(df):
    assert 0.0 <= proxy_check(df[config.ALLOWED], (df["career_gap_months"] > 0).astype(float))["auc"] <= 1.0
    assert np.isnan(proxy_check(df[config.ALLOWED], pd.Series([1.0] + [0.0] * (len(df) - 1), index=df.index))["auc"])


def test_protected_features_are_not_model_inputs(result):
    for m in result["fitted"].values():
        if hasattr(m, "named_steps"):
            names = " ".join(m.named_steps["prep"].get_feature_names_out())
            assert not re.search(r"gender|career_gap|age_|band_|hire_year|skills_count", names)
    assert "career_gap" not in " ".join(result["recommendations"].columns)
    assert not result["importance"]["feature"].str.contains(
        r"gender|career_gap|age_years|age_band|person_ref|record_type", case=False
    ).any()


def test_authorized_unlock_gender_audit_is_used_and_reported(result):
    status = result["summary"]["gender_audit"]
    assert status["run"] and status["matched_test_rows"] == len(result["test"])
    assert set(result["group_tables"]["gender"]["group"]) == set(status["gender_categories"])
    assert (result["group_tables"]["gender"].groupby("model")["n"].sum() == len(result["test"])).all()


def test_missing_unlock_makes_validation_incomplete(result):
    from cortexhr_task15.pipeline import validate_outputs

    incomplete = result.copy()
    incomplete["unlock_df"] = None
    incomplete["summary"] = result["summary"] | {
        "gender_audit": {"run": False, "required_file": "missing official unlock"}
    }
    checks = validate_outputs(incomplete)
    assert not next(c for c in checks if c["check"] == "official gender audit data loaded")["passed"]


def test_historical_gap_and_mitigation_tables_exist_and_keep_the_original(result):
    h = result["historical_gap"]["career_gap"]
    assert set(h["group"]) == {"Career gap 3-9 months", "Career gap 12+ months"} and (h["n_group"] > 0).all()
    m = result["mitigation"]
    assert m["variant"].iloc[0].startswith("M0 original") and len(m) == 3 and np.isfinite(m["mae"]).all()


# ------------------------------------------------------------------ salary bands
def test_bands_are_ordered_non_negative_and_centred_on_the_prediction():
    rng = np.random.RandomState(0)
    pred = rng.uniform(1e5, 2e6, 500); y = pred * rng.lognormal(0, 0.1, 500)
    cal = fit_calibration(pred, y)
    b = make_bands(pred, cal)
    assert (b["recommended_salary_low"] <= b["predicted_salary_mid"]).all() and (b["predicted_salary_mid"] <= b["recommended_salary_high"]).all() and (b >= 0).all().all()
    np.testing.assert_allclose(b["predicted_salary_mid"], np.round(pred))
    assert make_bands([-5.0], cal).iloc[0].min() >= 0                                      # a negative prediction cannot produce a negative salary


def test_band_calibration_uses_training_residuals_only(result):
    cal1 = result["calibration"]["global"]
    te = result["test"]
    changed = fit_level_calibration(np.ones(100), np.ones(100) * 2, np.ones(100))          # arbitrary other data gives different ratios, proving dependence on inputs only
    assert changed["global"].low_ratio != cal1.low_ratio
    assert cal1.n == len(result["train"])                                                  # calibrated on training out-of-fold residuals, one per training row


def test_band_coverage_is_measured_on_test_and_close_to_nominal(result):
    cov = result["summary"]["band"]["test_coverage"]
    assert 0.70 < cov["coverage"] < 0.90 and result["summary"]["band"]["nominal_level"] == pytest.approx(0.8)
    assert coverage([100.0], pd.DataFrame({"recommended_salary_low": [90], "predicted_salary_mid": [100], "recommended_salary_high": [110]}))["coverage"] == 1.0


def test_level_specific_bands_fall_back_when_a_level_is_small():
    pred, y, lv = np.ones(120) * 100, np.concatenate([np.ones(110) * 100, np.ones(10) * 500]), np.array([1] * 110 + [2] * 10)
    cals = fit_level_calibration(pred, y, lv, min_rows=40)
    assert cals["by_level"][2] is cals["global"]
    assert len(make_bands_by_level([100.0, 100.0], [1, 99], cals)) == 2                      # an unseen level also falls back


# ------------------------------------------------------------------ end to end
def test_validation_checks_all_pass(result):
    assert all(c["passed"] for c in result["validation"]), [c for c in result["validation"] if not c["passed"]]


def test_reports_and_models_written(result):
    d = result["dir"]
    for f in ["dataset_audit.json", "feature_policy.csv", "model_comparison.csv", "model_feature_importance.csv",
              "fairness_career_gap_groups.csv", "fairness_career_gap_official_group_gap.csv", "historical_pay_gap_controlled.csv", "mitigation_comparison.csv", "salary_band_recommendations.csv",
              "role_band_reference_2026.csv", "salary_distribution.png", "model_comparison.png", "fairness_gap.png", "final_validation_report.json"]:
        assert (d / f).exists() and (d / f).stat().st_size > 50, f
    for f in ("predictions_baseline.csv", "predictions_linear_regression.csv", "predictions_gradient_boosting.csv"):
        assert (d / "private" / f).exists() and (d / "private" / f).stat().st_size > 50
    assert len(list((d / "models").glob("*.joblib"))) == 3


def test_production_csv_schema_and_no_audit_data(result):
    rec = pd.read_csv(result["dir"] / "salary_band_recommendations.csv")
    assert list(rec.columns) == ["record_id", "role_id", "predicted_salary_mid", "recommended_salary_low", "recommended_salary_high", "model_version", "human_review_required", "model_note"]
    assert len(rec) == len(result["test"]) and rec["record_id"].is_unique and rec["human_review_required"].all() and (rec["model_version"] == config.MODEL_VERSION).all()
    assert not rec.isna().any().any() and "Final compensation must be reviewed by HR" in rec["model_note"].iloc[0]
    text = (result["dir"] / "salary_band_recommendations.csv").read_text().lower()
    assert "gender" not in text and "career_gap" not in text and "age_years" not in text


def test_gender_path_runs_end_to_end_with_a_synthetic_fixture_file(tmp_path, df):
    """SYNTHETIC random genders: tests the plumbing only. These numbers are never reported as results."""
    from cortexhr_task15 import config as c
    from cortexhr_task15.pipeline import run
    rng = np.random.RandomState(1)
    u = tmp_path / "unlock.csv"
    pd.DataFrame({"person_ref": df[c.ID_COL], "record_type": "x", "gender": rng.choice(["Male", "Female"], len(df)), "age_years": 30.0, "age_band": "25-34"}).to_csv(u, index=False)
    old = c.MODELS; c.MODELS = tmp_path / "m"
    try:
        r = run(reports=tmp_path / "r", unlock=u)
    finally:
        c.MODELS = old
    assert r["summary"]["gender_audit"]["run"] and "gender" in r["group_tables"] and (tmp_path / "r" / "fairness_gender_groups.csv").exists()
    assert not (tmp_path / "r" / "fairness_gender_NOT_RUN.md").exists()
    assert "gender" not in " ".join(r["recommendations"].columns) and all(x["passed"] for x in r["validation"])
