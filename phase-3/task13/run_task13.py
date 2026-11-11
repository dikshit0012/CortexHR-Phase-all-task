"""Task 13 - Attrition Model V2: run the whole pipeline end to end and write every report and the versioned artifact.

    python run_task13.py                      # from the 'task 13' folder
    python run_task13.py --version attrition-v2-rerun     # an existing version is never overwritten

Needs the master workbook (see README): data/raw/02_CortexHR_AI_ML_Dataset.xlsx under the Task 4 project,
or set CORTEXHR_WORKBOOK. Everything printed or written here is computed, none of it typed in.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from attrition_v2 import calibration as cal
from attrition_v2 import config, data_validation as dv, evaluation as ev
from attrition_v2 import hyperparameter_tuning as ht
from attrition_v2 import model_training as mt
from attrition_v2 import model_versioning as mv
from cortexhr.baselines import model_pipeline, load_model_table, per_fold_scores
from cortexhr.features import output_feature_names, select_features  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LR_BAL = "Logistic Regression (balanced) [Task 3 baseline]"
V2_DEFAULT = "LightGBM V2 (brief defaults, E00)"
V2_SAME_FOLD = "LightGBM tuned (config picked on the same folds; optimistic)"
V2_NESTED = "Tuned V2 (nested CV; honest)"


def lr_balanced():
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=config.RANDOM_STATE)


def main(version: str, save: bool = True) -> dict:
    t_start = time.time()
    R = config.REPORTS
    R.mkdir(exist_ok=True)
    out: dict = {}

    # ---------------------------------------------------------------- Phase 1/2: data + leakage audit
    print("[1/9] data and leakage audit")
    target = dv.target_column_in_workbook()
    X, y, source = dv.load_data()
    dv.assert_no_leakage_columns(X)
    audit = dv.dataset_audit()
    mv.write_json(audit, R / "data_audit.json")
    dv.excluded_columns_table().to_csv(R / "leakage_excluded_columns.csv", index=False)

    # ---------------------------------------------------------------- Phase 4: folds
    folds = dv.get_task3_folds(y)
    fold_check = dv.verify_folds(folds, y)
    assert fold_check["every_row_tested_once"] and not fold_check["train_test_overlap"]
    assert fold_check["identical_to_independent_StratifiedKFold_rebuild"]
    mv.write_json({**fold_check, "note": "Task 3 did not save fold indices; they are rebuilt with Task 3's own make_folds() "
                   "(StratifiedKFold(5, shuffle=True, random_state=42) over the 2,379 prepared rows).",
                   "folds": [{"fold": i, "train_idx": tr.tolist(), "test_idx": te.tolist()} for i, (tr, te) in enumerate(folds)]},
                  R / "cv_folds.json")

    scan = dv.single_feature_auc_scan(X, y, folds)
    scan.to_csv(R / "leakage_single_feature_auc.csv", index=False)
    shuffle_auc = dv.label_shuffle_control(lambda: mt.v2_pipeline(ht.full_params("E09")), X, y, folds)
    out["leakage"] = {"max_single_feature_cv_auc": float(scan["single_feature_cv_auc"].max()),
                      "top_single_feature": scan.iloc[0]["feature"], "label_shuffle_auc_v2": shuffle_auc,
                      "n_suspicious": int(scan["suspicious_gt_0.9"].sum())}
    print(f"      rows={len(y)} leavers={int(y.sum())} ({y.mean():.1%}); folds ok (fingerprint {fold_check['fold_fingerprint']}); "
          f"max single-feature AUC {out['leakage']['max_single_feature_cv_auc']:.3f}; label-shuffle AUC {shuffle_auc:.3f}")

    # ---------------------------------------------------------------- Phase 7: reference models on the same folds
    print("[2/9] reference models on the shared folds")
    oof: dict[str, np.ndarray] = mt.fit_all_reference(X, y, folds)
    kinds = {n: "reproduced_baseline" for n in oof}

    # ---------------------------------------------------------------- Phase 5/6: gradient boosting + tuning
    print("[3/9] LightGBM: brief defaults and the 13-experiment tuning log")
    log, tune_oofs = ht.run_tuning(X, y, folds)
    ht.tuning_log_for_csv(log).to_csv(R / "attrition_v2_tuning_log.csv", index=False)
    selected_id = str(log.loc[log["selected"], "id"].iloc[0])
    best_mean_id = str(log.loc[log["auc_mean"].idxmax(), "id"])
    oof[V2_DEFAULT] = tune_oofs["E00"]; kinds[V2_DEFAULT] = "new_model"
    oof[V2_SAME_FOLD] = tune_oofs[selected_id]; kinds[V2_SAME_FOLD] = "new_model"
    print(f"      selected {selected_id} (one-SE rule); highest mean was {best_mean_id}")

    print("[4/9] nested cross-validation (honest estimate of 'tune, then evaluate')")
    nested = ht.nested_cv(X, y, folds)
    oof[V2_NESTED] = nested["raw"]; kinds[V2_NESTED] = "new_model"
    nested["inner_tables"].to_csv(R / "attrition_v2_nested_cv_choices.csv", index=False)
    out["nested_choices"] = nested["chosen_per_fold"]

    # ---------------------------------------------------------------- Phase 7: comparison
    print("[5/9] same-fold comparison and official benchmarks")
    order = ["Dummy (always stays)", "Logistic Regression (unweighted)", LR_BAL, "Decision Tree (depth 4, balanced)",
             "Random Forest (300 trees, balanced)", V2_DEFAULT, V2_SAME_FOLD, V2_NESTED]
    comp = ev.compare_models(y, {k: oof[k] for k in order}, folds, kinds)
    comp.round(4).to_csv(R / "attrition_v2_comparison.csv", index=False)
    bench = ev.benchmark_comparison(comp, {
        "Dummy (always stays)": "Dummy (always stays)", LR_BAL: "Logistic Regression (class_weight=balanced)",
        "Decision Tree (depth 4, balanced)": "Decision Tree (depth 4, balanced)",
        "Random Forest (300 trees, balanced)": "Random Forest (300 trees, balanced)"})
    new_rows = comp[comp["kind"] == "new_model"]
    bench = pd.concat([bench, pd.DataFrame({
        "row_type": "new_model", "model": new_rows["model"], "roc_auc_mean": new_rows["roc_auc_mean"],
        "roc_auc_std": new_rows["roc_auc_std"], "pr_auc_mean": new_rows["pr_auc_mean"], "precision_mean": new_rows["precision_mean"],
        "recall_mean": new_rows["recall_mean"], "f1_mean": new_rows["f1_mean"], "accuracy_mean": new_rows["accuracy_mean"],
        "comment": "no official counterpart; compare with the official Logistic Regression / Random Forest rows"})],
        ignore_index=True)
    bench.round(4).to_csv(R / "attrition_v2_benchmark_comparison.csv", index=False)
    ev.per_source_auc(y, source, {k: oof[k] for k in order}).round(4).to_csv(R / "attrition_v2_per_source_auc.csv", index=False)

    paired = {
        "nested_V2_vs_LR_balanced": ev.paired_comparison(y, oof[V2_NESTED], oof[LR_BAL], folds),
        "same_fold_tuned_vs_LR_balanced": ev.paired_comparison(y, oof[V2_SAME_FOLD], oof[LR_BAL], folds),
        "brief_defaults_vs_LR_balanced": ev.paired_comparison(y, oof[V2_DEFAULT], oof[LR_BAL], folds),
        "nested_V2_vs_RandomForest": ev.paired_comparison(y, oof[V2_NESTED], oof["Random Forest (300 trees, balanced)"], folds),
        "nested_V2_vs_brief_defaults": ev.paired_comparison(y, oof[V2_NESTED], oof[V2_DEFAULT], folds)}
    mv.write_json(paired, R / "attrition_v2_paired_comparison.json")

    # sensitivity: the unconfirmed feature (open item O2). Reported only, never used for the decision.
    Xu, yu, _ = load_model_table(config.WORKBOOK, include_unconfirmed=True)
    assert (yu.to_numpy() == y.to_numpy()).all()
    sens = []
    def lgbm_pipe(inc):      # same Task 2 pipeline (with/without the unconfirmed feature) + the selected LightGBM config
        feats = model_pipeline(lr_balanced(), include_unconfirmed=inc).named_steps["features"]
        return Pipeline([("features", feats), ("model", mt.make_lgbm(ht.full_params(selected_id)))])
    for name, make in (("LR balanced", lambda inc: model_pipeline(lr_balanced(), include_unconfirmed=inc)),
                       (f"LightGBM {selected_id}", lgbm_pipe)):
        for inc, Xi in ((False, X), (True, Xu)):
            p = mt.oof_proba(make(inc), Xi, y, folds)
            pf = per_fold_scores(y, p, folds)
            sens.append({"model": name, "with_dept_attrition_rate_prior_12m": inc,
                         "roc_auc_mean": pf["roc_auc"].mean(), "roc_auc_sd": pf["roc_auc"].std(ddof=1)})
    pd.DataFrame(sens).round(4).to_csv(R / "sensitivity_unconfirmed_feature.csv", index=False)

    # ---------------------------------------------------------------- Phase 8: calibration
    print("[6/9] calibration (out-of-fold, calibrator fitted inside each training part)")
    cal_rows = {
        "LR balanced | none": ("Logistic Regression (balanced)", "none", oof[LR_BAL]),
        "LR balanced | sigmoid": ("Logistic Regression (balanced)", "sigmoid (Platt)",
                                  cal.calibrated_oof(lambda: model_pipeline(lr_balanced()), X, y, folds, "sigmoid")),
        "LR balanced | isotonic": ("Logistic Regression (balanced)", "isotonic",
                                   cal.calibrated_oof(lambda: model_pipeline(lr_balanced()), X, y, folds, "isotonic")),
        f"Tuned V2 (nested) | none": ("LightGBM tuned V2, class_weight=balanced", "none", nested["raw"]),
        f"Tuned V2 (nested) | sigmoid": ("LightGBM tuned V2, class_weight=balanced", "sigmoid (Platt)", nested["calibrated"]["sigmoid"]),
        f"Tuned V2 (nested) | isotonic": ("LightGBM tuned V2, class_weight=balanced", "isotonic", nested["calibrated"]["isotonic"]),
        f"LightGBM {selected_id}, unweighted | none": (
            "LightGBM (selected config), class_weight=None", "none",
            ht.score_config_on_folds(ht.full_params(selected_id), X, y, folds, class_weight=None)["oof"]),
    }
    cal_df = cal.calibration_comparison(cal_rows, y)
    cal_df.round(4).to_csv(R / "calibration_comparison.csv", index=False)
    cal.plot_calibration(y, {"Tuned V2 raw": nested["raw"], "Tuned V2 + sigmoid": nested["calibrated"]["sigmoid"],
                             "Tuned V2 + isotonic": nested["calibrated"]["isotonic"], "LR balanced raw": oof[LR_BAL],
                             "LR balanced + sigmoid": cal_rows["LR balanced | sigmoid"][2]},
                         R / "calibration_curve.png")

    def best_method(prefix: str) -> str:
        sub = cal_df[cal_df["label"].str.startswith(prefix) & ~cal_df["label"].str.endswith("none")].set_index("label")
        sig, iso = sub.loc[f"{prefix} | sigmoid"], sub.loc[f"{prefix} | isotonic"]
        # isotonic has more freedom (more variance with ~350 leavers): only prefer it if the Brier gain is clear
        return "isotonic" if iso["brier"] < sig["brier"] - 0.002 else "sigmoid"

    v2_cal_method, lr_cal_method = best_method("Tuned V2 (nested)"), best_method("LR balanced")
    get = lambda lbl: cal_df.set_index("label").loc[lbl]
    v2_cal_row, lr_cal_row = get(f"Tuned V2 (nested) | {v2_cal_method}"), get(f"LR balanced | {lr_cal_method}")

    # ---------------------------------------------------------------- Phase 9: decision
    print("[7/9] model selection")
    p_cal_pair = ev.paired_comparison(y, cal_rows[f"Tuned V2 (nested) | {v2_cal_method}"][2],
                                      cal_rows[f"LR balanced | {lr_cal_method}"][2], folds)
    decision = ev.selection_decision(paired["nested_V2_vs_LR_balanced"], v2_cal_row["ece_quantile_10bins"],
                                     lr_cal_row["ece_quantile_10bins"])
    decision["paired_vs_LR_balanced"] = paired["nested_V2_vs_LR_balanced"]
    decision["paired_calibrated_vs_calibrated"] = p_cal_pair
    ship_gb = decision["ship_gradient_boosting"]
    print(f"      ship gradient boosting: {ship_gb}  checks={decision['checks']}")

    # ---------------------------------------------------------------- Phase 10: final model on all rows
    print("[8/9] final model and artifact")
    if ship_gb:
        family, method = "LightGBM", v2_cal_method
        cfg_id, cfg = selected_id, ht.full_params(selected_id)
        base = lambda: mt.v2_pipeline(cfg)
        hyper = {"classifier": "LGBMClassifier", "class_weight": "balanced", **cfg, "random_state": config.RANDOM_STATE,
                 "config_id": cfg_id}
        proba_oof = cal_rows[f"Tuned V2 (nested) | {v2_cal_method}"][2]
    else:
        family, method = "Logistic Regression", lr_cal_method
        cfg_id = None
        base = lambda: model_pipeline(lr_balanced())
        hyper = {"classifier": "LogisticRegression", "class_weight": "balanced", "max_iter": 2000,
                 "random_state": config.RANDOM_STATE}
        proba_oof = cal_rows[f"LR balanced | {lr_cal_method}"][2]
    final = CalibratedClassifierCV(estimator=base(), method=method,
                                   cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE + 2)).fit(X, y)
    final_oof_metrics = ev.compare_models(y, {"final": proba_oof}, folds).iloc[0].to_dict()
    final_cal = cal.calibration_metrics(y, proba_oof)

    # importance of the boosted model on all rows (leakage sanity check, reported either way)
    fi = mt.feature_importance(mt.v2_pipeline(ht.full_params(selected_id)).fit(X, y))
    fi.round(4).to_csv(R / "attrition_v2_feature_importance.csv", index=False)

    metadata = {
        "model_name": "CortexHR attrition model V2", "model_family": family, "selected_by": "rule in evaluation.selection_decision",
        "target_column": target, "prediction_horizon": "left within 12 months of the observation date",
        "feature_list": list(X.columns), "n_features_in": X.shape[1],
        "excluded_columns": dv.excluded_columns_table()["column"].tolist(),
        "preprocessing": "Task 2 pipeline cortexhr.features.build_pipeline() (RawCleaner + median impute/scale + one-hot), "
                         "fitted inside the saved estimator; monthly attendance/payroll features must be built with "
                         "cortexhr.monthly.add_monthly_features before scoring",
        "hyperparameters": hyper, "calibration_method": method,
        "calibration_note": "CalibratedClassifierCV, 5 stratified inner folds, ensemble of 5 calibrated models",
        "cross_validation": {"strategy": "Task 3 folds: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
                             "fold_fingerprint": fold_check["fold_fingerprint"],
                             "tuning": "nested CV on the same outer folds; configuration choice by one-standard-error rule"},
        "random_seed": config.RANDOM_STATE,
        "dataset": {"workbook": Path(config.WORKBOOK).name, "sheet": "attrition_dataset", "rows_raw": audit["raw_shape"][0],
                    "rows_trained": int(len(y)), "leavers": int(y.sum()), "row_rules": audit["row_rules"]},
        "gradient_boosting_challenger_shipped": bool(ship_gb),
        "decision": decision,
    }
    metrics = {"evaluation": "out-of-fold on the 5 Task 3 folds; mean/sd are per-fold (yes/no metrics at threshold 0.5 on the "
                             "calibrated probability)",
               "final_model_oof": final_oof_metrics, "final_model_calibration_oof": final_cal,
               "baseline_LR_balanced_oof": comp.set_index("model").loc[LR_BAL].to_dict(),
               "tuned_V2_nested_raw_oof": comp.set_index("model").loc[V2_NESTED].to_dict(),
               "paired_comparison_vs_LR_balanced": paired["nested_V2_vs_LR_balanced"]}
    folder = None
    if save:
        folder = mv.save_model(final, metadata, metrics, version=version)
        print(f"      saved {folder}")

    # ---------------------------------------------------------------- Phase 11: prediction check on the unlabeled scoring sheet
    print("[9/9] prediction check on attrition_scoring_2026 (new records, no label)")
    pred_summary = None
    if save:
        loaded = mv.load_model(version)
        scoring = mv.build_scoring_table()
        preds = mv.predict_attrition(scoring, loaded)
        pred_summary = {"n_scored": int(len(preds)), "p_min": float(preds.iloc[:, 0].min()), "p_max": float(preds.iloc[:, 0].max()),
                        "p_mean": float(preds.iloc[:, 0].mean()), "model_version": loaded.version}
        print("      ", pred_summary)

    out.update({"audit": audit, "folds": fold_check, "tuning_selected": selected_id, "tuning_best_mean": best_mean_id,
                "decision": decision, "final_family": family, "final_calibration": method, "version": version,
                "artifact": str(folder) if folder else None, "prediction_check": pred_summary,
                "runtime_seconds": round(time.time() - t_start, 1)})
    mv.write_json(out, R / "task13_run_summary.json")
    print(f"done in {out['runtime_seconds']} s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=config.MODEL_VERSION)
    ap.add_argument("--no-save", action="store_true", help="run everything except writing the model artifact")
    a = ap.parse_args()
    main(a.version, save=not a.no_save)
