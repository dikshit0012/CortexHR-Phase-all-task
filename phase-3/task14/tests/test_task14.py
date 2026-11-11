"""Task 14 tests: model, scoring, bands, SHAP, reasons, sensitive-data rules, output schema, edge cases."""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cortexhr_task14 import config
if config.locate_cortexhr_src() is None:
    pytest.skip("Task 14 tests require the Task 2-4 cortexhr package (included at phase-1/task04/src) or CORTEXHR_ML_SRC.", allow_module_level=True)

task13_root = Path(os.environ.get("CORTEXHR_TASK13_ROOT", config.TASK14_ROOT.parent / "task13"))
if not (task13_root / "model_artifacts" / "attrition-v2" / "model.joblib").is_file():
    pytest.skip("Task 14 tests require the Task 13 attrition-v2 model artifact.", allow_module_level=True)

from cortexhr_task14 import reason_generator as rg
from cortexhr_task14.employee_scoring import ScoringDataError, validate_features, validate_ids, validate_probabilities
from cortexhr_task14.model_loader import ModelVerificationError, verify_model
from cortexhr_task14.output_formatter import build_output, model_note, pick_sample, read_sample_schema
from cortexhr_task14.risk_band_assignment import assign_risk_bands, verify_bands
from cortexhr_task14.validation import validate_output


# ------------------------------------------------------------------ model
def test_model_loads_and_is_attrition_v2(loaded):
    info = verify_model(loaded)
    assert info["model_version"] == "attrition-v2" and info["estimator"] == "CalibratedClassifierCV"
    assert info["ensemble_members"] == 5 and info["n_features"] == 33 and info["classes"] == [0, 1]


def test_wrong_version_is_rejected(loaded):
    import copy
    bad = copy.copy(loaded)
    bad.metadata = {**loaded.metadata, "version": "attrition-v1"}
    with pytest.raises(ModelVerificationError):
        verify_model(bad)


def test_feature_order_matches_training(loaded, inputs):
    assert list(inputs.X.columns) == loaded.required_columns == loaded.metadata["feature_list"]


# ------------------------------------------------------------------ scoring
def test_430_employees_scored_with_valid_probabilities(result):
    s = result["scores"]
    assert len(s) == 430 and s["employee_id"].is_unique and s["attrition_probability"].between(0, 1).all()
    assert (s["model_version"] == "attrition-v2").all() and s["attrition_probability"].notna().all()


def test_scores_align_with_employees_and_use_positive_class(result, loaded):
    X, meta, s = result["X"], result["meta"], result["scores"]
    p_all = loaded.model.predict_proba(X)
    assert list(s["employee_id"]) == list(meta["employee_id"])
    np.testing.assert_allclose(s["attrition_probability"], p_all[:, 1], atol=1e-15)           # column 1 = leaves
    assert not np.allclose(s["attrition_probability"], p_all[:, 0])                            # not the 'stays' probability
    one = X.iloc[[7]]
    assert loaded.model.predict_proba(one)[0, 1] == pytest.approx(s["attrition_probability"].iloc[7])   # row alignment


def test_employee_id_validation():
    validate_ids(pd.Series([f"E{i}" for i in range(430)]))
    with pytest.raises(ScoringDataError):
        validate_ids(pd.Series(["E1", None, "E3"]), expected=3)                 # missing id
    with pytest.raises(ScoringDataError):
        validate_ids(pd.Series(["E1", "E1", "E3"]), expected=3)                 # duplicate
    with pytest.raises(ScoringDataError):
        validate_ids(pd.Series(["E1", "E2"]), expected=430)                      # wrong count (never manufactured)
    with pytest.raises(ScoringDataError):
        validate_ids(pd.Series(["E1", "  ", "E3"]), expected=3)                 # blank id


def test_feature_validation(loaded, inputs):
    assert isinstance(validate_features(inputs.X, loaded.required_columns), dict)
    with pytest.raises(ScoringDataError):
        validate_features(inputs.X.drop(columns=["comp_ratio"]), loaded.required_columns)
    with pytest.raises(ScoringDataError):
        validate_features(inputs.X.assign(extra_col=1), loaded.required_columns)            # no silent extra columns
    with pytest.raises(ScoringDataError):
        validate_features(inputs.X[list(reversed(inputs.X.columns))], loaded.required_columns)


@pytest.mark.parametrize("bad", [[0.1, np.nan], [0.1, 1.2], [-0.1, 0.2], [np.inf], []])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ScoringDataError):
        validate_probabilities(bad)


def test_missing_feature_values_do_not_crash_scoring(loaded, inputs):
    X = inputs.X.iloc[:5].copy()
    for c in ("last_engagement_score", "comp_ratio", "work_mode", "department_id"):
        X.loc[X.index[0], c] = np.nan
    p = loaded.model.predict_proba(X)[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


# ------------------------------------------------------------------ risk bands
def test_bands_come_from_official_helper(result):
    from cortexhr_task14.cortexhr_helpers import risk_bands
    p = result["scores"]["attrition_probability"].to_numpy()
    assert list(result["bands"]) == list(risk_bands(p))
    assert result["summary"]["risk_bands"]["counts"] == {"Low": 301, "Medium": 86, "High": 43} or sum(result["summary"]["risk_bands"]["counts"].values()) == 430


def test_band_rules_and_boundaries():
    p = np.linspace(0.01, 0.50, 100)
    b = assign_risk_bands(p)
    assert (b == "High").sum() == 10 and (b == "Medium").sum() == 20 and (b == "Low").sum() == 70
    assert verify_bands(p, b)["consistent"]
    t = pd.Series(p).quantile(0.90)
    assert assign_risk_bands(np.array(list(p) + [t])).iloc[-1] == "High"                 # value equal to the cut-off goes up (>=)
    with pytest.raises(ScoringDataError):
        assign_risk_bands([0.1, np.nan, 0.3])                                         # the helper alone would label NaN 'Low'
    with pytest.raises(ScoringDataError):
        verify_bands([0.1, 0.9], ["Low", "Bogus"])


# ------------------------------------------------------------------ SHAP
def test_shap_additivity_on_the_real_model_output(result):
    s = result["shap"]
    assert s.values.shape == (430, 33) and list(s.values.columns) == result["loaded"].required_columns
    assert s.additivity_error.max() < 1e-6
    np.testing.assert_allclose(s.predictions, result["scores"]["attrition_probability"], atol=1e-12)      # same predictions as Phase 4
    assert s.values.notna().all().all() and np.isfinite(s.values.to_numpy()).all()


def test_shap_small_real_subset_is_deterministic_and_additive(loaded, inputs):
    from cortexhr_task14.shap_explainer import AttritionExplainer
    sub = inputs.X.iloc[:3]
    a = AttritionExplainer(loaded, inputs.X, nsamples=300).explain(sub)
    b = AttritionExplainer(loaded, inputs.X, nsamples=300).explain(sub)
    pd.testing.assert_frame_equal(a.values, b.values)                               # seeded -> reproducible
    assert a.additivity_error.max() < 1e-6


def test_encode_decode_round_trip_reproduces_predictions(loaded, inputs):
    from cortexhr_task14.shap_explainer import AttritionExplainer
    ex = AttritionExplainer(loaded, inputs.X, nsamples=100)
    D = ex.decode(ex.encode(inputs.X))
    np.testing.assert_allclose(loaded.model.predict_proba(D)[:, 1], loaded.model.predict_proba(inputs.X)[:, 1], atol=1e-12)


def test_shap_direction_matches_model_behaviour(result):
    """A feature with a clearly positive contribution should, when set to its typical value, usually LOWER the model's prediction."""
    d = result["summary"]["counterfactual_direction_check"]
    assert d["employees_checked"] > 50 and d["share"] > 0.85


# ------------------------------------------------------------------ reasons: selection and direction
def _vals(**kw):
    base = {f: 1.0 for f in rg.FEATURE_POLICY}
    base.update({"department_id": "D01", "role_id": "R001", "work_mode": "Hybrid", "employment_type": "Permanent", "work_location": "Pune", "highest_education": "MBA"})
    base.update(kw)
    return pd.Series(base)


def _pop(n=60):
    rng = np.random.RandomState(0)
    pop = pd.DataFrame({f: rng.rand(n) * 10 for f in rg.FEATURE_POLICY})
    for f in ("department_id", "role_id", "work_mode", "employment_type", "work_location", "highest_education"):
        pop[f] = "x"
    return pop


def _contrib(**kw):
    c = pd.Series(0.0, index=list(rg.FEATURE_POLICY))
    for k, v in kw.items():
        c[k] = v
    return c


def test_top_three_are_the_strongest_risk_raising_drivers_in_order():
    c = _contrib(last_engagement_score=0.05, comp_ratio=0.03, avg_overtime_hours_per_month_3m=0.02, months_since_last_raise=0.01, training_hours_12m=-0.09)
    r = rg.select_reasons(c, _vals(), _pop())
    assert [x.feature for x in r["reasons"]] == ["last_engagement_score", "comp_ratio", "avg_overtime_hours_per_month_3m"]
    assert all(x.direction == "raises" and x.contribution > 0 for x in r["reasons"])
    assert r["n_risk_raising"] == 3                                                  # the big NEGATIVE value is not picked as a risk driver


def test_largest_absolute_value_is_not_chosen_when_it_lowers_risk():
    c = _contrib(training_hours_12m=-0.30, last_engagement_score=0.02)
    r = rg.select_reasons(c, _vals(), _pop())
    assert r["reasons"][0].feature == "last_engagement_score" and r["reasons"][0].direction == "raises"
    assert r["reasons"][1].feature == "training_hours_12m" and r["reasons"][1].direction == "lowers"
    assert "lower-than-average predicted attrition risk" in r["reasons"][1].text


def test_fewer_than_three_drivers_is_reported_honestly_not_invented():
    r = rg.select_reasons(_contrib(comp_ratio=0.02), _vals(), _pop())
    assert [x.direction for x in r["reasons"]] == ["raises", "none", "none"] and r["n_empty"] == 2
    assert r["reasons"][1].text == rg.NO_FURTHER and r["reasons"][1].feature is None


def test_zero_and_sub_threshold_contributions_are_never_named():
    r = rg.select_reasons(_contrib(comp_ratio=0.0, last_engagement_score=config.MATERIALITY / 2, job_level=-config.MATERIALITY / 2), _vals(), _pop())
    assert r["n_empty"] == 3 and all(x.feature is None for x in r["reasons"])


def test_positive_and_negative_contributions_keep_their_direction():
    pos = rg.build_reason(rg.FEATURE_POLICY["comp_ratio"], 0.8, 1, _pop()["comp_ratio"])
    neg = rg.build_reason(rg.FEATURE_POLICY["comp_ratio"], 1.2, -1, _pop()["comp_ratio"])
    assert "higher-than-average predicted attrition risk" in pos and "lower-than-average predicted attrition risk" in neg
    assert rg.build_reason(rg.FEATURE_POLICY["comp_ratio"], 0.8, 0) is None


def test_no_theme_is_repeated_so_reasons_cannot_contradict():
    c = _contrib(raises_received=-0.04, months_since_last_raise=0.05, last_raise_pct=0.03, comp_ratio=0.02, last_engagement_score=0.01)
    r = rg.select_reasons(c, _vals(), _pop())
    themes = [x.theme for x in r["reasons"] if x.theme]
    assert len(themes) == len(set(themes)) and "raises" in themes and r["reasons"][0].feature == "months_since_last_raise"


def test_unknown_feature_name_is_an_error():
    with pytest.raises(KeyError):
        rg.select_reasons(pd.Series({"mystery_feature": 0.3}), pd.Series({"mystery_feature": 1}), pd.DataFrame())


def test_missing_values_named_only_when_the_gap_is_an_hr_fact():
    c = _contrib(months_since_last_promotion=0.03, last_engagement_score=0.04, comp_ratio=0.02)
    v = _vals(months_since_last_promotion=np.nan, last_engagement_score=np.nan)
    r = rg.select_reasons(c, v, _pop())
    feats = [x.feature for x in r["reasons"]]
    assert "months_since_last_promotion" in feats and "last_engagement_score" not in feats
    assert "last_engagement_score" in r["data_gap_features"]
    text = next(x.text for x in r["reasons"] if x.feature == "months_since_last_promotion")
    assert text.startswith("No promotion is recorded since joining")


# ------------------------------------------------------------------ wording, tenure, sensitive data
def test_tenure_wording_is_neutral_in_both_directions():
    for d in (1, -1):
        t = rg.build_reason(rg.FEATURE_POLICY["tenure_months"], 5, d, _pop()["tenure_months"])
        assert rg.TENURE_NOTE in t and "likely to leave" not in t and "short tenure" not in t.lower() and "new employee" not in t.lower()


def test_no_protected_or_proxy_feature_can_be_named():
    assert set(rg.EXCLUDED) >= {"total_experience_years", "highest_education", "work_location", "commute_distance_km", "unpaid_leave_days_6m",
                                "paid_leave_days_6m", "avg_unpaid_absence_days_3m"}
    c = _contrib(total_experience_years=0.5, highest_education=0.4, work_location=0.3, paid_leave_days_6m=0.2, comp_ratio=0.01)
    r = rg.select_reasons(c, _vals(), _pop())
    assert [x.feature for x in r["reasons"]][0] == "comp_ratio" and r["suppressed_abs_contribution"] == pytest.approx(1.4)
    for f in rg.EXCLUDED:
        assert rg.build_reason(rg.FEATURE_POLICY[f], 5, 1, None) is None
    assert not [f for f in rg.FEATURE_POLICY if re.search(r"gender|\bage\b|dob|birth|religion|caste|name|photo", f)]


def test_every_model_feature_has_a_policy(loaded):
    assert set(rg.FEATURE_POLICY) == set(loaded.required_columns)


def test_audit_text_catches_sensitive_and_certain_wording():
    assert rg.audit_text("The employee is a woman and will leave") and rg.audit_text("He is a flight risk") and rg.audit_text("Definitely quitting")
    assert rg.audit_text("Hello Priya Sharma", ["Priya Sharma"])
    assert not rg.audit_text("Recorded pay is contributing to a higher predicted attrition risk.")


def test_department_reason_is_labelled_group_level():
    t = rg.build_reason(rg.FEATURE_POLICY["department_id"], "D08", 1)
    assert "group-level pattern" in t and "not a finding about this individual" in t


# ------------------------------------------------------------------ output
def test_exactly_430_records_matching_the_official_sample(result):
    out = result["output"]
    assert len(out) == 430 and list(out.columns) == config.SAMPLE_COLUMNS == read_sample_schema(config_workbook())
    assert out["employee_id"].is_unique and out["prediction_id"].is_unique


def config_workbook():
    from cortexhr_task14.model_loader import mv
    return mv.config.WORKBOOK


def test_all_validation_checks_pass(result):
    failed = [c for c in result["summary"]["validation"] if not c["passed"]]
    assert not failed, failed


def test_human_review_required_on_every_row(result):
    out, internal = result["output"], result["internal"]
    assert out["model_note"].str.contains("human_review_required = true").all() and out["model_note"].str.contains(re.escape(config.DISCLAIMER)).all()
    assert internal["human_review_required"].all()
    assert "Every result requires human review before any HR action." in model_note()


def test_csv_files_written_and_round_trip(result):
    d = result["dir"]
    for f in ("attrition_risk_explanations.csv", "attrition_risk_sample_10.csv", "attrition_shap_summary.png", "attrition_shap_bar.png"):
        assert (d / f).exists() and (d / f).stat().st_size > 1000, f
    back = pd.read_csv(d / "attrition_risk_explanations.csv")
    assert len(back) == 430 and list(back.columns) == config.SAMPLE_COLUMNS and not back.isna().any().any()
    s10 = pd.read_csv(d / "attrition_risk_sample_10.csv")
    assert len(s10) == 10 and set(s10["risk_band"]) == {"Low", "Medium", "High"} and s10["employee_id"].isin(back["employee_id"]).all()
    assert (d / "internal" / "attrition_scoring_internal.csv").exists() and (d / "internal" / "validation_report.json").exists()


def test_internal_columns_stay_out_of_the_official_csv(result):
    assert not {"shap_base_value", "human_review_required", "attrition_probability"} & set(result["output"].columns)
    assert {"shap_base_value", "human_review_required", "shap__comp_ratio"} <= set(result["internal"].columns)


def test_sample_10_is_representative_and_deterministic(result):
    a, b = pick_sample(result["output"]), pick_sample(result["output"])
    pd.testing.assert_frame_equal(a, b)
    assert a["risk_band"].value_counts().to_dict() == {"High": 4, "Medium": 3, "Low": 3}


def test_validation_catches_broken_outputs(result):
    ids = list(result["meta"]["employee_id"])
    good = result["output"]
    assert all(c["passed"] for c in validate_output(good, ids))
    def failed(df):
        return {c["check"] for c in validate_output(df, ids) if not c["passed"]}
    assert "exactly 430 employees" in failed(good.iloc[:-1])
    assert "no duplicate employee ids" in failed(pd.concat([good.iloc[:-1], good.iloc[[0]]], ignore_index=True))
    assert "risk scores numeric and within [0, 1]" in failed(good.assign(risk_score_12m=good["risk_score_12m"] + 1))
    assert "risk bands valid" in failed(good.assign(risk_band="Critical"))
    assert "model version is attrition-v2" in failed(good.assign(model_version="attrition-v1"))
    assert "no protected / sensitive terms, certainty claims or names in reasons" in failed(good.assign(top_reason_1="This woman will leave."))
    assert "human review requirement + mandatory notice present on every row" in failed(good.assign(model_note="ok"))
    assert "columns and order match the official sample" in failed(good[list(reversed(good.columns))])
    assert "no empty / NaN reason cells" in failed(good.assign(top_reason_2=np.nan))


def test_explanations_are_individual_and_varied(result):
    out = result["output"]
    assert out["top_reason_1"].nunique() > 100
    assert result["importance"].iloc[0]["mean_abs_shap"] > 0 and result["importance"]["share_of_total"].sum() == pytest.approx(1.0)


def test_reasons_use_only_allowed_features_and_no_names(result):
    used = pd.concat([result["internal"][f"reason_{i}_feature"] for i in (1, 2, 3)]).dropna()
    assert set(used) <= set(rg.ELIGIBLE)
    ids = " ".join(result["output"]["employee_id"])
    text = " ".join(result["output"][["top_reason_1", "top_reason_2", "top_reason_3"]].stack())
    assert "EMP-" not in text and not rg.audit_text(text)


# ------------------------------------------------------------------ wording regressions found in the manual review
def test_singular_and_plural_units():
    assert "1 month)" in rg.build_reason(rg.FEATURE_POLICY["tenure_months"], 1, 1) and "12 months)" in rg.build_reason(rg.FEATURE_POLICY["tenure_months"], 12, 1)


def test_typical_values_are_not_described_as_typical_and_no_double_brackets():
    pop = pd.Series(np.arange(100.0))
    mid = rg.build_reason(rg.FEATURE_POLICY["job_level"], 50.0, 1, pop)
    low = rg.build_reason(rg.FEATURE_POLICY["job_level"], 3.0, 1, pop)
    assert "typical" not in mid and "which is on the lower side compared with other active employees," in low and ")(" not in low and ") (" not in low


def test_extreme_attendance_change_is_flagged_as_a_data_check_not_a_finding():
    t = rg.build_reason(rg.FEATURE_POLICY["attendance_rate_change_3m"], 0.454, 1)
    assert "may be worth checking" in t and "+45.4 percentage points" in t
    assert "worth checking" not in rg.build_reason(rg.FEATURE_POLICY["attendance_rate_change_3m"], -0.05, 1)


def test_note_says_some_inputs_are_never_named():
    assert "deliberately never named" in model_note()


# ------------------------------------------------------------------ Task 13 integrity and the total_experience_years policy
def test_task13_artifact_and_source_are_unchanged():
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify13", config.TASK14_ROOT / "scripts" / "verify_task13_unchanged.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.verify()
    assert r["unchanged"], r
    assert r["model_joblib_sha256"] == json.loads((config.TASK14_ROOT / "docs" / "task13_sha256.json").read_text())["files"]["model_artifacts/attrition-v2/model.joblib"]


def test_total_experience_years_is_suppressed_but_still_a_model_input(loaded, result):
    assert "total_experience_years" in loaded.required_columns                       # the model still uses it ...
    assert "total_experience_years" in rg.EXCLUDED and not rg.FEATURE_POLICY["total_experience_years"].eligible
    assert "total_experience_years" in result["shap"].values.columns                   # ... and its SHAP values are computed ...
    used = pd.concat([result["internal"][f"reason_{i}_feature"] for i in (1, 2, 3)]).dropna()
    assert "total_experience_years" not in set(used)                                  # ... but it is never named
    assert result["importance"].set_index("feature").loc["total_experience_years", "named_in_reasons"] == False  # noqa: E712


def test_suppressing_a_feature_does_not_change_the_scores(result, loaded):
    """Explanation policy is post-processing: the scores (and bands) come from the model, which still receives the feature."""
    X = result["X"]
    p = loaded.model.predict_proba(X)[:, 1]
    np.testing.assert_allclose(result["scores"]["attrition_probability"], p, atol=1e-15)
    Z = X.copy()
    Z["total_experience_years"] = X["total_experience_years"].median()
    assert np.abs(loaded.model.predict_proba(Z)[:, 1] - p).max() > 0.01                # the feature really moves predictions
