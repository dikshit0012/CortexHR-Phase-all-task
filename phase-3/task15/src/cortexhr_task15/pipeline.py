"""End-to-end Task 15. Order (follows the official workflow): audit data -> split -> select/fit models on TRAIN only -> evaluate on the untouched TEST set
-> build bands from training residuals -> ONLY THEN open the audit attributes (career gap from the table, gender from unlock_CX-15) for fairness tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from . import config, plots
from .bands import coverage, fit_level_calibration, make_bands_by_level
from .bias_analysis import historical_pay_gap, mitigation_variants
from .data import audit, load_salary_training, validate
from .fairness import (attach_gender, career_gap_groups, gap_summary, group_table, load_unlock, proxy_check)
from .models import BASELINE, BOOSTING, LINEAR, NAMES, build_model, cv_oof, feature_names, metrics, split

MODELS = (BASELINE, LINEAR, BOOSTING)
PROD_COLUMNS = ["record_id", "role_id", "predicted_salary_mid", "recommended_salary_low", "recommended_salary_high", "model_version", "human_review_required", "model_note"]


def _j(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


def _feature_importance(model) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    names = feature_names(model)
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        method = f"{type(estimator).__name__}.feature_importances_"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
        method = "absolute coefficient magnitude"
    else:
        raise TypeError(f"Unsupported estimator for feature importance: {type(estimator).__name__}")
    return pd.DataFrame({"feature": names, "importance": values, "method": method}).sort_values(
        "importance", ascending=False, ignore_index=True
    )


def run(workbook: Optional[Path] = None, reports: Optional[Path] = None, unlock: Optional[Path] = None, write: bool = True) -> Dict[str, Any]:
    reports = Path(reports) if reports else config.REPORTS
    df = load_salary_training(workbook)
    basic = validate(df)
    aud = audit(df)
    train, test = split(df)
    Xtr, Xte, ytr, yte = train[config.ALLOWED], test[config.ALLOWED], train[config.TARGET], test[config.TARGET]

    # ---- model selection on TRAIN only (5-fold out-of-fold), then one evaluation on the untouched TEST set
    oof = {m: cv_oof(m, Xtr, ytr) for m in MODELS}
    cv = {m: {"cv_mae": float(np.mean(np.abs(oof[m] - ytr))), "cv_r2": float(1 - ((oof[m] - ytr) ** 2).sum() / ((ytr - ytr.mean()) ** 2).sum())} for m in MODELS}
    fitted = {m: build_model(m).fit(Xtr, ytr) for m in MODELS}
    pred = {m: fitted[m].predict(Xte) for m in MODELS}
    test_m = {m: metrics(yte, pred[m]) for m in MODELS}
    train_m = {m: metrics(ytr, fitted[m].predict(Xtr)) for m in MODELS}
    selected = min((LINEAR, BOOSTING), key=lambda m: cv[m]["cv_mae"])             # decided from training-fold results only
    importance = _feature_importance(fitted[selected])
    comparison = pd.DataFrame([{"Model": NAMES[m], "MAE": test_m[m]["mae"], "R2": test_m[m]["r2"], "CV_MAE_train_only": cv[m]["cv_mae"], "CV_R2_train_only": cv[m]["cv_r2"],
                                "train_MAE": train_m[m]["mae"], "train_R2": train_m[m]["r2"], "selected": m == selected} for m in MODELS])

    # ---- salary bands (training residuals only; test labels only measure coverage)
    calibrations = {
        model_name: fit_level_calibration(oof[model_name], ytr, train["job_level"].to_numpy())
        for model_name in MODELS
    }
    cal = calibrations[selected]
    bands = make_bands_by_level(pred[selected], test["job_level"].to_numpy(), cal)
    cov = coverage(yte, bands)
    cov_by_level = {int(l): coverage(yte[test["job_level"] == l], bands[(test["job_level"] == l).to_numpy()]) | {"n": int((test["job_level"] == l).sum())} for l in sorted(test["job_level"].unique())}
    rec = pd.DataFrame({"record_id": test[config.ID_COL].to_numpy(), "role_id": test["role_id"].to_numpy()})
    rec = pd.concat([rec, bands[["predicted_salary_mid", "recommended_salary_low", "recommended_salary_high"]]], axis=1)
    rec["model_version"], rec["human_review_required"], rec["model_note"] = config.MODEL_VERSION, True, config.NOTE
    rec = rec[PROD_COLUMNS]
    role_ref = _role_reference(df, fitted[selected], cal)

    # ---- audit attributes opened only now
    band_ref = (df["band_mid_2026"] * df["market_index"])
    te_ref = band_ref.loc[test.index]
    groups_cg = {"career_gap": career_gap_groups(test["career_gap_months"])}
    unlock_df = load_unlock(unlock)
    expected_unlock = str(Path(unlock)) if unlock else f"one of {config.UNLOCK_NAMES} in {config.PRIVATE_DIR}"
    gender_status = {"run": unlock_df is not None, "required_file": expected_unlock}
    groupings = {"career_gap": groups_cg["career_gap"]}
    if unlock_df is not None:
        te_g = attach_gender(test, unlock_df)["gender"]
        gender_status.update({
            "matched_test_rows": int(te_g.notna().sum()),
            "unmatched_test_rows": int(te_g.isna().sum()),
            "gender_counts_test": te_g.value_counts(dropna=False).to_dict(),
            "gender_categories": sorted(str(v) for v in unlock_df["gender"].dropna().unique()),
            "unlock_rows": len(unlock_df),
        })
        groupings["gender"] = te_g

    def frame(m):
        model_bands = make_bands_by_level(pred[m], test["job_level"].to_numpy(), calibrations[m])
        f = pd.DataFrame({"actual": yte.to_numpy(float), "predicted": pred[m], "band_ref": te_ref.to_numpy(float),
                          "rec_low": model_bands["recommended_salary_low"].to_numpy(),
                          "rec_mid": model_bands["predicted_salary_mid"].to_numpy(),
                          "rec_high": model_bands["recommended_salary_high"].to_numpy()})
        for k, v in groupings.items():
            f[k] = v.to_numpy()
        return f
    gtabs, gsums = {}, {}
    for k in groupings:
        gtabs[k] = pd.concat([group_table(frame(m), k, NAMES[m]) for m in MODELS], ignore_index=True)
        gsums[k] = pd.concat([gap_summary(frame(m), k, NAMES[m]) for m in MODELS], ignore_index=True)

    # ---- bias investigation
    hist = {"career_gap": historical_pay_gap(df.assign(_g=career_gap_groups(df["career_gap_months"])), career_gap_groups(df["career_gap_months"]), "No career gap")}
    proxy = {"career_gap_from_allowed_features_auc": proxy_check(df[config.ALLOWED], (df["career_gap_months"] > 0).astype(float))}
    if unlock_df is not None:
        g_all = attach_gender(df, unlock_df)["gender"]
        hist["gender"] = historical_pay_gap(df, g_all, "Male" if "Male" in set(g_all.dropna()) else g_all.mode().iloc[0])
        proxy["gender_female_vs_nonfemale_from_allowed_features_auc"] = proxy_check(
            df[config.ALLOWED], (g_all == "Female").astype(float).where(g_all.notna())
        )
    mitig = mitigation_variants(train, test, {k: v for k, v in groupings.items()}, te_ref)
    bias = {"historical_salary_in_features": [c for c in config.ALLOWED if "salary" in c.lower() or "prior_pay" in c.lower()], "gender_in_features": [c for c in config.ALLOWED if "gender" in c.lower()],
            "career_gap_in_features": [c for c in config.ALLOWED if "gap" in c.lower()], "proxy_checks": proxy,
            "historical_pay_difference_controlled_for_job_factors": {k: v.to_dict("records") for k, v in hist.items()}, "gender_audit": gender_status}

    summary = {"data": basic | {"audit": aud}, "split": {"strategy": "random 80/20 stratified by source; hire IDs are unique, but no separate stable employee identifier is available to rule out repeat-person leakage", "seed": config.SEED,
               "train": len(train), "test": len(test)}, "model_comparison": comparison.to_dict("records"), "selected_model": selected, "non_positive_test_predictions": {m: int((pred[m] <= 0).sum()) for m in MODELS}, "selection_rule": "lowest 5-fold out-of-fold MAE on TRAIN rows between Linear Regression and Gradient Boosting",
               "band": {"method": "Mid = prediction; Low/High = Mid x 10th/90th percentile of (actual/predicted) in training out-of-fold residuals, per job level", "global_low_ratio": cal["global"].low_ratio, "global_high_ratio": cal["global"].high_ratio, "ratios_by_level": {l: [c.low_ratio, c.high_ratio, c.n] for l, c in cal["by_level"].items()}, "nominal_level": cal["global"].high_q - cal["global"].low_q, "test_coverage": cov, "test_coverage_by_job_level": cov_by_level},
               "bias_investigation": bias, "features": {"allowed": config.ALLOWED, "audit_only": config.AUDIT_ONLY},
               "feature_importance": importance.head(20).to_dict("records"), "gender_audit": gender_status, "recommendations": len(rec)}
    res = {"df": df, "train": train, "test": test, "pred": pred, "comparison": comparison, "bands": bands, "recommendations": rec, "role_reference": role_ref, "group_tables": gtabs, "gap_summaries": gsums,
           "historical_gap": hist, "mitigation": mitig, "summary": summary, "fitted": fitted, "calibration": cal, "selected": selected, "audit": aud,
           "importance": importance, "unlock_df": unlock_df}
    res["validation"] = validate_outputs(res)
    summary["validation"] = res["validation"]
    summary["all_checks_passed"] = all(c["passed"] for c in res["validation"])
    if write:
        _write(res, reports)
    return res


def _role_reference(df: pd.DataFrame, model, cal) -> pd.DataFrame:
    """Typical-profile band per role at the April-2026 price level (market_index = 1): modal department / level / location / education, median experience."""
    rows = []
    for role, g in df.groupby("role_id"):
        rows.append({"role_id": role, "department_id": g["department_id"].mode().iloc[0], "job_level": int(g["job_level"].mode().iloc[0]), "prior_experience_years": float(g["prior_experience_years"].median()),
                     "work_location": g["work_location"].mode().iloc[0], "highest_education": g["highest_education"].mode().iloc[0], "market_index": 1.0, "n_training_hires": len(g)})
    r = pd.DataFrame(rows)
    b = make_bands_by_level(model.predict(r[config.ALLOWED]), r["job_level"].to_numpy(), cal)
    out = pd.concat([r[["role_id", "department_id", "job_level", "n_training_hires"]], b], axis=1)
    out["model_version"], out["human_review_required"] = config.MODEL_VERSION, True
    return out


def validate_outputs(res: Dict[str, Any]) -> list:
    rec, test, comp = res["recommendations"], res["test"], res["comparison"]
    c = lambda n, ok, e="": {"check": n, "passed": bool(ok), "evidence": e}
    out = [
        c("recommendation rows = held-out test rows", len(rec) == len(test), f"{len(rec)} / {len(test)}"),
        c("no duplicate record ids", not rec["record_id"].duplicated().any(), int(rec["record_id"].duplicated().sum())),
        c("no missing recommendations", not rec[["predicted_salary_mid", "recommended_salary_low", "recommended_salary_high"]].isna().any().any()),
        c("no negative salaries", (rec[["predicted_salary_mid", "recommended_salary_low", "recommended_salary_high"]] >= 0).all().all()),
        c("low <= mid <= high for every row", ((rec["recommended_salary_low"] <= rec["predicted_salary_mid"]) & (rec["predicted_salary_mid"] <= rec["recommended_salary_high"])).all()),
        c("role association preserved", (rec["role_id"].to_numpy() == test["role_id"].to_numpy()).all()),
        c("model version correct", (rec["model_version"] == config.MODEL_VERSION).all(), config.MODEL_VERSION),
        c("human review flagged on every row", rec["human_review_required"].all()),
        c("columns equal the production schema", list(rec.columns) == PROD_COLUMNS, list(rec.columns)),
        c("no protected / audit attribute in the production CSV", not (set(rec.columns) & config.SENSITIVE_NEVER_FEATURES)),
        c("no protected attribute in the predictive feature set", not (set(config.ALLOWED) & (config.SENSITIVE_NEVER_FEATURES | {"band_min_2026", "band_mid_2026", "band_max_2026", config.TARGET})), config.ALLOWED),
        c("metrics finite for all three models", np.isfinite(comp[["MAE", "R2"]].to_numpy()).all()),
        c("train and test rows are disjoint", not (set(res["train"][config.ID_COL]) & set(test[config.ID_COL]))),
        c("official gender audit data loaded", res["unlock_df"] is not None, res["summary"]["gender_audit"].get("required_file")),
        c("gender audit covers test records", res["summary"]["gender_audit"].get("matched_test_rows", 0) > 0
          and res["summary"]["gender_audit"].get("unmatched_test_rows", len(test)) == 0),
        c("gender fairness results generated", "gender" in res["group_tables"] and "gender" in res["gap_summaries"]),
        c("model feature importance generated", len(res["importance"]) > 0),
        c("band ratios are ordered and positive (every level)", all(0 < x.low_ratio <= 1 <= x.high_ratio for x in [res["calibration"]["global"], *res["calibration"]["by_level"].values()])),
        c("role reference bands ordered", (res["role_reference"]["recommended_salary_low"] <= res["role_reference"]["predicted_salary_mid"]).all() and (res["role_reference"]["predicted_salary_mid"] <= res["role_reference"]["recommended_salary_high"]).all()),
    ]
    return out


def _write(res: Dict[str, Any], reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    private_reports = reports / "private"
    private_reports.mkdir(exist_ok=True)
    prediction_files = (
        "predictions_baseline.csv",
        "predictions_linear_regression.csv",
        "predictions_gradient_boosting.csv",
    )
    for filename in prediction_files:
        legacy_path = reports / filename
        if legacy_path.exists():
            legacy_path.replace(private_reports / filename)
    fmt = dict(index=False, float_format="%.6g")
    test, pred = res["test"], res["pred"]
    pd.DataFrame(res["audit"]["dtypes"].items(), columns=["column", "dtype"]).to_csv(reports / "dataset_columns.csv", index=False)
    (reports / "dataset_audit.json").write_text(json.dumps(res["audit"], indent=2, default=_j), encoding="utf-8")
    pol = pd.DataFrame([{"column": k, "decision": v[0], "reason": v[1]} for k, v in config.POLICY.items()])
    pol.to_csv(reports / "feature_policy.csv", index=False)
    res["comparison"].to_csv(reports / "model_comparison.csv", **fmt)
    res["importance"].to_csv(reports / "model_feature_importance.csv", **fmt)
    for m, fn in ((BASELINE, "predictions_baseline"), (LINEAR, "predictions_linear_regression"), (BOOSTING, "predictions_gradient_boosting")):
        pd.DataFrame({"hire_id": test[config.ID_COL].to_numpy(), "role_id": test["role_id"].to_numpy(), "actual": test[config.TARGET].to_numpy(), "predicted": pred[m],
                      "error": pred[m] - test[config.TARGET].to_numpy()}).to_csv(private_reports / f"{fn}.csv", **fmt)
    res["recommendations"].to_csv(reports / "salary_band_recommendations.csv", index=False)
    res["role_reference"].to_csv(reports / "role_band_reference_2026.csv", index=False)
    for k, t in res["group_tables"].items():
        (reports / f"fairness_{k}_groups.csv").write_text(t.to_csv(index=False, float_format="%.6g"))
        res["gap_summaries"][k].to_csv(reports / f"fairness_{k}_official_group_gap.csv", **fmt)
    if "gender" not in res["group_tables"]:
        (reports / "fairness_gender_NOT_RUN.md").write_text(
            "# Gender fairness audit: NOT RUN\n\n"
            f"The required official `unlock_CX-15` file was not found at {res['summary']['gender_audit']['required_file']}. "
            "No gender findings were fabricated. Pipeline validation fails until the authorized file is supplied and the pipeline is rerun.\n",
            encoding="utf-8",
        )
    else:
        stale_status = reports / "fairness_gender_NOT_RUN.md"
        if stale_status.exists():
            stale_status.unlink()
    pd.concat([v.assign(grouping=k) for k, v in res["historical_gap"].items()]).to_csv(reports / "historical_pay_gap_controlled.csv", **fmt)
    res["mitigation"].to_csv(reports / "mitigation_comparison.csv", **fmt)
    plots.salary_distribution(res["df"][config.TARGET], res["bands"], reports / "salary_distribution.png")
    plots.model_comparison(res["comparison"], reports / "model_comparison.png")
    sel = NAMES[res["selected"]]
    plots.fairness_gaps({f"{k} - {sel}": t[t["model"] == sel] for k, t in res["group_tables"].items()}, reports / "fairness_gap.png")
    config.MODELS.mkdir(parents=True, exist_ok=True)
    for m, mod in res["fitted"].items():
        joblib.dump(mod, config.MODELS / f"{config.MODEL_VERSION}_{m}.joblib")
    (reports / "final_validation_report.json").write_text(json.dumps(res["summary"], indent=2, default=_j), encoding="utf-8")
