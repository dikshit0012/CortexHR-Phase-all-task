"""End-to-end Task 14: load attrition-v2 -> score 430 employees -> official bands -> SHAP -> reasons -> CSVs, charts, validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import config
from .employee_scoring import build_scoring_inputs, score_employees
from .model_loader import load_attrition_v2, verify_model
from .output_formatter import (build_output, feature_importance, internal_report, pick_sample, plot_bar, plot_summary, read_sample_schema)
from .reason_generator import ELIGIBLE, policy_table, select_reasons
from .risk_band_assignment import assign_risk_bands, verify_bands
from .shap_explainer import AttritionExplainer, ShapResult
from .validation import counterfactual_direction_check, validate_output


def _fingerprint(preds: np.ndarray, settings: Dict) -> str:
    return hashlib.sha256((np.round(preds, 12).tobytes().hex() + json.dumps(settings, sort_keys=True)).encode()).hexdigest()[:16]


def compute_shap(loaded, X: pd.DataFrame, cache_dir: Path, recompute: bool = False, progress: bool = True) -> ShapResult:
    """Real SHAP values; cached by a fingerprint of (predictions, settings) so a re-run with identical inputs is instant and identical."""
    ex = AttritionExplainer(loaded, X)
    preds = ex._predict(ex.encode(X))
    settings = {"background": ex.background_index.tolist(), "nsamples": ex.nsamples, "seed": ex.seed, "cols": ex.columns}
    fp = _fingerprint(preds, settings)
    f_vals, f_meta = cache_dir / "shap_values.csv", cache_dir / "shap_values.meta.json"
    if not recompute and f_vals.exists() and f_meta.exists() and json.loads(f_meta.read_text())["fingerprint"] == fp:
        vals = pd.read_csv(f_vals, index_col=0)
        vals.index = X.index
        err = np.abs(ex.base_value + vals.sum(axis=1).to_numpy() - preds)
        return ShapResult(vals, ex.base_value, preds, err, {**json.loads(f_meta.read_text())["settings"], "from_cache": True})
    res = ex.explain(X, progress=progress)
    cache_dir.mkdir(parents=True, exist_ok=True)
    res.values.to_csv(f_vals, float_format="%.10f")
    f_meta.write_text(json.dumps({"fingerprint": fp, "settings": res.settings}, indent=2))
    return res


def run(workbook=None, reports_dir: Optional[Path] = None, recompute_shap: bool = False, write: bool = True, stability_check: bool = True,
        progress: bool = True) -> Dict[str, Any]:
    reports_dir = Path(reports_dir) if reports_dir else config.REPORTS
    internal_dir = reports_dir / "internal"
    loaded = load_attrition_v2()
    model_info = verify_model(loaded)
    inputs = build_scoring_inputs(loaded, workbook)
    X, meta = inputs.X, inputs.meta
    scores = score_employees(inputs, loaded)
    bands = assign_risk_bands(scores["attrition_probability"])
    band_info = verify_bands(scores["attrition_probability"], bands)
    wb = workbook or __import__("attrition_v2").model_versioning.config.WORKBOOK
    schema = read_sample_schema(wb)

    shap_res = compute_shap(loaded, X, internal_dir, recompute_shap, progress)
    selections = [select_reasons(shap_res.values.iloc[i], X.iloc[i], X) for i in range(len(X))]
    reasons = [[r.text for r in s["reasons"]] for s in selections]
    out = build_output(meta, scores, bands, reasons)
    internal = internal_report(meta, scores, bands, shap_res, selections, X)
    checks = validate_output(out, list(meta["employee_id"]), schema, internal=internal)
    direction = counterfactual_direction_check(loaded, X, internal, None)
    stab = None
    if stability_check:
        from .validation_stability import stability_check as _sc
        stab = _sc(loaded, X, shap_res)
    imp = feature_importance(shap_res.values)
    sample10 = pick_sample(out)
    summary = {"model": model_info, "employees": len(out), "risk_bands": band_info, "missing_values_in_scoring_data": inputs.missing_report,
               "shap": shap_res.settings, "shap_base_value": shap_res.base_value, "shap_max_additivity_error": float(shap_res.additivity_error.max()),
               "probability_summary": scores["attrition_probability"].describe().to_dict(),
               "reasons": {"employees_with_all_three_reasons": int((internal["n_empty_reason_slots"] == 0).sum()),
                           "employees_with_fewer": int((internal["n_empty_reason_slots"] > 0).sum()),
                           "employees_with_3_risk_raising_reasons": int((internal["n_risk_raising_reasons"] == 3).sum()),
                           "employees_with_0_risk_raising_reasons": int((internal["n_risk_raising_reasons"] == 0).sum()),
                           "materiality_threshold": config.MATERIALITY},
               "top_features": imp.head(10)[["feature", "mean_abs_shap_percentage_points", "named_in_reasons"]].to_dict("records"),
               "counterfactual_direction_check": direction, "explanation_stability": stab,
               "validation": checks, "all_checks_passed": all(c["passed"] for c in checks)}
    result = {"output": out, "sample10": sample10, "internal": internal, "importance": imp, "summary": summary, "shap": shap_res, "X": X, "meta": meta,
              "scores": scores, "bands": bands, "loaded": loaded, "selections": selections}
    if write:
        reports_dir.mkdir(parents=True, exist_ok=True)
        internal_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(reports_dir / "attrition_risk_explanations.csv", index=False)
        if (config.TASK14_ROOT / "attrition_risk_explanations.csv").exists() and (reports_dir != config.TASK14_ROOT):
            out.to_csv(config.TASK14_ROOT / "attrition_risk_explanations.csv", index=False)
        sample10.to_csv(reports_dir / "attrition_risk_sample_10.csv", index=False)
        plot_summary(shap_res.values, X, reports_dir / "attrition_shap_summary.png")
        plot_bar(shap_res.values, reports_dir / "attrition_shap_bar.png")
        internal.to_csv(internal_dir / "attrition_scoring_internal.csv", index=False, float_format="%.12g")
        imp.to_csv(internal_dir / "shap_feature_importance.csv", index=False, float_format="%.12g")
        policy_table().to_csv(internal_dir / "feature_policy.csv", index=False)
        (internal_dir / "validation_report.json").write_text(json.dumps(summary, indent=2, default=_j), encoding="utf-8")
    return result


def _j(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)
