"""Task 13 tests. They use the real workbook (see README) and the artifact written by ``run_task13.py``."""
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

default_ml_src = Path(__file__).resolve().parents[2] / "task 4" / "files" / "cortexhr-ml" / "src"
ml_src = Path(os.environ.get("CORTEXHR_ML_SRC", default_ml_src))
if not (ml_src / "cortexhr").is_dir():
    pytest.skip("Task 13 tests require the separately supplied Task 2-4 cortexhr package.", allow_module_level=True)

from attrition_v2 import calibration as cal                                   # noqa: E402
from attrition_v2 import config, data_validation as dv, evaluation as ev      # noqa: E402
from attrition_v2 import hyperparameter_tuning as ht, model_training as mt    # noqa: E402
from attrition_v2 import model_versioning as mv                               # noqa: E402
from cortexhr.baselines import make_folds, model_pipeline                     # noqa: E402
from cortexhr.features import ForbiddenColumnError                            # noqa: E402
from sklearn.linear_model import LogisticRegression                           # noqa: E402
from sklearn.metrics import roc_auc_score                                     # noqa: E402

REPORTS = config.REPORTS


@pytest.fixture(scope="module")
def data():
    if not config.WORKBOOK.is_file():
        pytest.skip(f"Task 13 data tests require the authorized workbook at {config.WORKBOOK}.")
    return dv.load_data()


@pytest.fixture(scope="module")
def folds(data):
    return dv.get_task3_folds(data[1])


@pytest.fixture(scope="module")
def loaded():
    if not (config.ARTIFACTS / config.MODEL_VERSION / "model.joblib").is_file():
        pytest.skip("Task 13 model tests require the locally generated attrition-v2 artifact.")
    return mv.load_model()


# 1 ---------------------------------------------------------------------------------------------
def test_dataset_loading(data):
    X, y, source = data
    assert X.shape == (2379, 33)                     # 2,389 rows - 6 duplicates - 4 negative tenure; 33 safe inputs
    assert len(y) == len(source) == len(X)
    assert int(y.sum()) == 349 and abs(y.mean() - 0.1467) < 1e-3


# 2 ---------------------------------------------------------------------------------------------
def test_target_validation(data):
    assert dv.target_column_in_workbook() == "left_within_12m"
    dv.validate_target(data[1])
    with pytest.raises(dv.TargetError):
        dv.validate_target(pd.Series([0, 1, 2]))
    with pytest.raises(dv.TargetError):
        dv.validate_target(pd.Series([0, 1, np.nan]))
    with pytest.raises(dv.TargetError):
        dv.validate_target(pd.Series([], dtype=int))


# 3 ---------------------------------------------------------------------------------------------
def test_leakage_prone_features_excluded(data):
    X = data[0]
    dv.assert_no_leakage_columns(X)
    table = dv.excluded_columns_table()
    for c in ["left_within_12m", "exit_date", "resignation_date", "exit_reason", "exit_interview_done",
              "fnf_settlement_amount", "voluntary_left_within_12m", "dept_attrition_rate_prior_12m", "employee_id"]:
        assert c in set(table["column"]) and c not in X.columns
    for bad in ["exit_date", "left_within_12m", "voluntary_left_within_12m"]:
        with pytest.raises(ForbiddenColumnError):
            dv.assert_no_leakage_columns(X.assign(**{bad: 0}))
    with pytest.raises(ValueError):
        dv.assert_no_leakage_columns(X.assign(dept_attrition_rate_prior_12m=0.1))


def test_no_single_feature_looks_like_a_leak():
    scan = pd.read_csv(REPORTS / "leakage_single_feature_auc.csv")
    assert scan["single_feature_cv_auc"].max() < 0.7 and not scan["suspicious_gt_0.9"].any()
    shuffle = json.loads((REPORTS / "task13_run_summary.json").read_text())["leakage"]["label_shuffle_auc_v2"]
    assert 0.40 < shuffle < 0.60


# 4 ---------------------------------------------------------------------------------------------
def test_preprocessing_is_fitted_on_training_rows_only(data, folds):
    X, y, _ = data
    tr, te = folds[0]
    est = mt.v2_pipeline(ht.full_params("E09")).fit(X.iloc[tr], y.iloc[tr])
    feats = est.named_steps["features"]
    Z_te = feats.transform(X.iloc[te])
    assert Z_te.shape[0] == len(te) and np.isfinite(Z_te).all()
    full = model_pipeline(LogisticRegression()).named_steps["features"].fit(X)
    scaler_train = feats.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    scaler_full = full.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    assert not np.allclose(scaler_train.mean_, scaler_full.mean_)      # statistics depend on the rows it was fitted on
    assert scaler_train.n_samples_seen_ == len(tr)


# 5 ---------------------------------------------------------------------------------------------
def test_identical_cross_validation_folds(data, folds):
    X, y, _ = data
    check = dv.verify_folds(folds, y)
    assert check["n_folds"] == 5 and check["seed"] == 42 and check["every_row_tested_once"]
    assert not check["train_test_overlap"] and check["identical_to_independent_StratifiedKFold_rebuild"]
    assert dv.fold_fingerprint(folds) == dv.fold_fingerprint(make_folds(y))      # Task 3's own function
    assert max(check["fold_positive_rates"]) - min(check["fold_positive_rates"]) < 0.005     # stratified
    saved = json.loads((REPORTS / "cv_folds.json").read_text())
    assert saved["fold_fingerprint"] == dv.fold_fingerprint(folds)               # reports were produced with these folds
    for i, (tr, te) in enumerate(folds):
        assert saved["folds"][i]["test_idx"] == te.tolist()


# 6 ---------------------------------------------------------------------------------------------
def test_model_training_is_valid_and_deterministic(data, folds):
    X, y, _ = data
    tr, te = folds[0]
    p1 = mt.v2_pipeline(ht.full_params("E09")).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    p2 = mt.v2_pipeline(ht.full_params("E09")).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    assert np.array_equal(p1, p2)
    assert p1.min() >= 0 and p1.max() <= 1 and roc_auc_score(y.iloc[te], p1) > 0.6
    assert mt.make_lgbm().get_params()["random_state"] == config.RANDOM_STATE


# 7 ---------------------------------------------------------------------------------------------
def test_one_standard_error_rule_prefers_the_simpler_model():
    t = pd.DataFrame({"id": ["A", "B", "C"], "auc_mean": [0.72, 0.719, 0.70], "auc_se": [0.01, 0.01, 0.01],
                      "complexity": [1000, 100, 10]})
    assert ht.one_se_select(t) == "B"            # within 1 SE of the best and simpler; C is simplest but too weak


def test_hyperparameter_tuning_and_log(data, folds):
    X, y, _ = data
    two = [c for c in ht.CONFIGS if c["id"] in ("E02", "E09")]
    log, oofs = ht.run_tuning(X, y, folds, configs=two)
    assert set(oofs) == {"E02", "E09"} and log["selected"].sum() == 1
    assert (log["train_auc_mean"] >= log["auc_mean"]).all()
    csv = pd.read_csv(REPORTS / "attrition_v2_tuning_log.csv")
    assert len(csv) == len(ht.CONFIGS) == 13 and csv["selected"].sum() == 1
    for col in ["experiment_id", "model", "hyperparameters", "cv_roc_auc_mean", "cv_pr_auc_mean", "cv_f1_mean",
                "cv_precision_mean", "cv_recall_mean", "notes"]:
        assert col in csv.columns
    row = csv[csv["experiment_id"] == "E09"].iloc[0]                     # same numbers as a fresh run on the same folds
    assert abs(row["cv_roc_auc_mean"] - log.set_index("id").loc["E09", "auc_mean"]) < 1e-4


def test_nested_cv_never_uses_outer_test_rows_for_choices(data):
    X, y, _ = data
    sub = np.arange(0, len(y), 3)
    Xs, ys = X.iloc[sub].reset_index(drop=True), y.iloc[sub].reset_index(drop=True)
    from sklearn.model_selection import StratifiedKFold
    f = list(StratifiedKFold(3, shuffle=True, random_state=config.RANDOM_STATE).split(Xs, ys))
    out = ht.nested_cv(Xs, ys, f, configs=[c for c in ht.CONFIGS if c["id"] in ("E02", "E09")], methods=("sigmoid",),
                       inner_splits=3)
    assert len(out["chosen_per_fold"]) == 3 and set(out["chosen_per_fold"]) <= {"E02", "E09"}
    assert np.all((out["raw"] >= 0) & (out["raw"] <= 1)) and np.all((out["calibrated"]["sigmoid"] >= 0))


# 8 ---------------------------------------------------------------------------------------------
def test_evaluation_metrics_are_correct(data, folds):
    _, y, _ = data
    perfect = ev.compare_models(y, {"perfect": y.to_numpy().astype(float)}, folds).iloc[0]
    assert perfect["roc_auc_mean"] == 1 and perfect["precision_mean"] == 1 and perfect["recall_mean"] == 1
    const = ev.compare_models(y, {"const": np.zeros(len(y))}, folds).iloc[0]
    assert const["roc_auc_mean"] == 0.5 and const["f1_mean"] == 0
    assert abs(const["accuracy_mean"] - (1 - y.mean())) < 0.01
    rng = np.random.default_rng(0)
    p = rng.random(len(y))
    same = ev.paired_comparison(y, p, p, folds, n_boot=200)
    assert same["roc_auc_diff_mean"] == 0 and same["folds_won"] == 0
    comp = pd.read_csv(REPORTS / "attrition_v2_comparison.csv")
    lr = comp[comp["model"].str.contains("Task 3 baseline")].iloc[0]
    assert abs(lr["roc_auc_mean"] - 0.712) < 0.002                          # reproduces Task 3 on the same folds


def test_official_benchmark_values_come_from_the_workbook():
    bench = pd.read_csv(REPORTS / "attrition_v2_benchmark_comparison.csv")
    official = bench[bench["row_type"] == "official_benchmark"].set_index("model")
    assert abs(official.loc["Logistic Regression (class_weight=balanced)", "roc_auc_mean"] - 0.713) < 1e-9
    assert set(bench["row_type"]) == {"official_benchmark", "reproduced_baseline", "new_model"}
    from cortexhr.baselines import load_benchmarks
    assert len(official) == len(load_benchmarks())


def test_selection_rule():
    good = {"roc_auc_diff_mean": 0.02, "pooled_diff_ci95": [0.005, 0.03], "folds_won": 5}
    assert ev.selection_decision(good, 0.02, 0.02)["ship_gradient_boosting"]
    uncertain = {"roc_auc_diff_mean": 0.02, "pooled_diff_ci95": [-0.01, 0.05], "folds_won": 5}
    assert not ev.selection_decision(uncertain, 0.02, 0.02)["ship_gradient_boosting"]
    small = {"roc_auc_diff_mean": 0.005, "pooled_diff_ci95": [0.001, 0.01], "folds_won": 5}
    assert not ev.selection_decision(small, 0.02, 0.02)["ship_gradient_boosting"]
    assert not ev.selection_decision(good, 0.08, 0.02)["ship_gradient_boosting"]      # worse calibration


# 9 ---------------------------------------------------------------------------------------------
def test_calibration_metrics_detect_over_confidence():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0.02, 0.5, 6000)
    y = (rng.random(6000) < p_true).astype(int)
    good = cal.calibration_metrics(y, p_true)
    assert good["ece_quantile_10bins"] < 0.03 and 0.8 < good["calibration_slope"] < 1.2
    over = cal.calibration_metrics(y, np.clip(p_true * 2.5, 0, 1))
    assert over["ece_quantile_10bins"] > good["ece_quantile_10bins"] + 0.1 and over["brier"] > good["brier"]
    with pytest.raises(ValueError):
        cal.calibration_metrics(y, p_true * 5)


def test_calibration_improves_balanced_logistic_regression(data, folds):
    X, y, _ = data
    factory = lambda: model_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
    raw = mt.oof_proba(factory(), X, y, folds)
    sig = cal.calibrated_oof(factory, X, y, folds, "sigmoid")
    m_raw, m_sig = cal.calibration_metrics(y, raw), cal.calibration_metrics(y, sig)
    assert m_sig["ece_quantile_10bins"] < m_raw["ece_quantile_10bins"] / 3 and m_sig["brier"] < m_raw["brier"]
    assert abs(m_sig["mean_predicted"] - y.mean()) < 0.01
    assert abs(m_sig["roc_auc"] - m_raw["roc_auc"]) < 0.02                   # ranking essentially unchanged
    saved = pd.read_csv(REPORTS / "calibration_comparison.csv").set_index("label")
    assert abs(saved.loc["LR balanced | sigmoid", "brier"] - m_sig["brier"]) < 1e-3
    assert (REPORTS / "calibration_curve.png").stat().st_size > 10_000


# 10 --------------------------------------------------------------------------------------------
def test_model_serialization(tmp_path, data):
    X, y, _ = data
    est = model_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    folder = mv.save_model(est, {"feature_list": list(X.columns)}, {"x": 1}, version="attrition-vtest", root=tmp_path)
    for f in ("model.joblib", "metadata.json", "metrics.json"):
        assert (folder / f).is_file()
    assert json.loads((folder / "metadata.json").read_text())["version"] == "attrition-vtest"
    with pytest.raises(FileExistsError):                                       # versions are never overwritten
        mv.save_model(est, {"feature_list": []}, {}, version="attrition-vtest", root=tmp_path)
    again = mv.load_model("attrition-vtest", root=tmp_path)
    assert np.allclose(again.model.predict_proba(X.iloc[:20]), est.predict_proba(X.iloc[:20]))


# 11 --------------------------------------------------------------------------------------------
def test_model_loading(loaded):
    meta = loaded.metadata
    assert loaded.version == "attrition-v2" and meta["target_column"] == "left_within_12m"
    assert len(loaded.required_columns) == 33 and meta["random_seed"] == 42
    for key in ["model_name", "version", "trained_at_utc", "target_column", "feature_list", "preprocessing", "hyperparameters",
                "cross_validation", "calibration_method", "random_seed", "dataset", "library_versions", "decision"]:
        assert key in meta, key
    assert not set(meta["feature_list"]) & set(meta["excluded_columns"])
    assert "final_model_oof" in loaded.metrics and hasattr(loaded.model, "predict_proba")


# 12 + 13 + 14 ----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def new_records():
    return mv.build_scoring_table().head(60)


def test_prediction_on_new_records_and_probability_range(loaded, new_records, caplog):
    with caplog.at_level(logging.INFO, logger="attrition_v2"):
        out = mv.predict_attrition(new_records, loaded)
    assert out.shape == (60, 2) and list(out.columns) == ["attrition_probability_12m", "model_version"]
    p = out["attrition_probability_12m"]
    assert p.between(0, 1).all() and p.notna().all() and p.nunique() > 10
    assert (out["model_version"] == "attrition-v2").all() and out.index.equals(new_records.index)
    assert all("EMP-" not in r.getMessage() for r in caplog.records)           # no employee values in logs
    assert any("scored 60 records" in r.getMessage() for r in caplog.records)


def test_missing_values_are_handled_but_missing_columns_are_not(loaded, new_records):
    holes = new_records.copy()
    holes.iloc[:10, :] = np.nan
    holes["department_id"] = holes["department_id"].astype(object)
    out = mv.predict_attrition(holes, loaded)
    assert out["attrition_probability_12m"].between(0, 1).all()
    with pytest.raises(mv.ModelInputError):
        mv.predict_attrition(new_records.drop(columns=["tenure_months"]), loaded)
    with pytest.raises(mv.ModelInputError):
        mv.predict_attrition(new_records.iloc[0:0], loaded)
    with pytest.raises(ForbiddenColumnError):
        mv.predict_attrition(new_records.assign(exit_date="2026-01-01"), loaded)
    with pytest.raises(mv.ModelInputError):
        mv.predict_attrition([1, 2, 3], loaded)


def test_saved_pipeline_matches_training_preprocessing(loaded, data):
    X, y, _ = data
    p = mv.predict_attrition(X.head(300), loaded)["attrition_probability_12m"]
    assert roc_auc_score(y.head(300), p) > 0.6                               # in-sample, only a "wiring is right" check
    shuffled_cols = X.head(300)[list(reversed(X.columns))]
    assert np.allclose(mv.predict_attrition(shuffled_cols, loaded)["attrition_probability_12m"], p)   # columns by name


def test_model_version_identification(loaded):
    assert config.MODEL_VERSION == "attrition-v2"
    assert loaded.folder.name == "attrition-v2" and loaded.metadata["version"] == "attrition-v2"
    with pytest.raises(FileNotFoundError):
        mv.load_model("attrition-v999")
