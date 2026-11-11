"""Cross-check that README.md, docs/FINAL_AUDIT.md, reports/internal/*.json and the CSV outputs agree. Exit code 1 on any disagreement.
Hand-written README numbers are compared with values recomputed from the CSVs / JSON; generated blocks are compared with regenerated text."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, I, D = ROOT / "reports", ROOT / "reports" / "internal", ROOT / "docs"
sys.path.insert(0, str(ROOT / "src"))
from cortexhr_task14 import config  # noqa: E402

readme = (ROOT / "README.md").read_text(encoding="utf-8")
audit = (D / "FINAL_AUDIT.md").read_text(encoding="utf-8")
val = json.loads((I / "validation_report.json").read_text())
exp = json.loads((I / "total_experience_analysis.json").read_text())
tests = json.loads((I / "test_summary.json").read_text())
out = pd.read_csv(R / "attrition_risk_explanations.csv")
s10 = pd.read_csv(R / "attrition_risk_sample_10.csv")
internal = pd.read_csv(I / "attrition_scoring_internal.csv")
imp = pd.read_csv(I / "shap_feature_importance.csv")
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def has(text, s):
    return s in text


# --- CSV vs JSON vs internal
check("430 rows in the official CSV, JSON and internal report", len(out) == val["employees"] == len(internal) == 430)
check("official CSV columns/order equal the sample schema", list(out.columns) == config.SAMPLE_COLUMNS)
check("band counts: CSV == JSON", out["risk_band"].value_counts().to_dict() == val["risk_bands"]["counts"], val["risk_bands"]["counts"])
check("internal bands == CSV bands", (internal["risk_band"].values == out["risk_band"].values).all())
check("internal probability rounds to CSV score", (internal["attrition_probability"].round(3).values == out["risk_score_12m"].values).all())
check("sample_10 rows are verbatim rows of the final CSV", s10.merge(out, how="left", indicator=True, on=list(out.columns))["_merge"].eq("both").all())
check("all model_version attrition-v2 (CSV, internal)", (out["model_version"] == "attrition-v2").all() and (internal["model_version"] == "attrition-v2").all() and val["model"]["model_version"] == "attrition-v2")
check("human review on every row (note + internal flag)", out["model_note"].str.contains("human_review_required = true").all() and internal["human_review_required"].all())
check("JSON additivity error matches the internal report", abs(val["shap_max_additivity_error"] - internal["shap_additivity_error"].max()) < 1e-12)
ct = val["risk_bands"]; c = out["risk_band"].value_counts()
check("validation JSON: every check passed", val["all_checks_passed"] and all(x["passed"] for x in val["validation"]), f"{len(val['validation'])} checks")
check("test summary: no failures and Task 13 unchanged", tests["failed"] == 0 and tests["passed"] > 0 and tests["task13_unchanged"], tests["pytest_summary_line"])

# --- hand-written README numbers recomputed from the data
p = out["risk_score_12m"]; pr = pd.Series(internal["attrition_probability"])
facts = {
    "band counts High/Medium/Low": f"**High {c['High']}, Medium {c['Medium']}, Low {c['Low']}**",
    "helper cut-offs": f"**{ct['high_cutoff_p90']:.4f}** and **{ct['medium_cutoff_p70']:.4f}**",
    "probability range/mean/median": f"**{pr.min():.3f} - {pr.max():.3f}, mean {pr.mean():.3f}, median {pr.median():.3f}**",
    "SHAP base value": f"**{val['shap_base_value']:.4f}**",
    "reasons: 3 raising": f"**{(internal['n_risk_raising_reasons'] == 3).sum()} employees have three risk-raising reasons, {(internal['n_risk_lowering_reasons'] > 0).sum()} have at least one risk-lowering reason, {(internal['n_risk_raising_reasons'] == 0).sum()} has no risk-raising reason, {int(internal['n_empty_reason_slots'].sum())} slots are empty.**",
    "stability": f"**{val['explanation_stability']['identical_top3_sets']}/{val['explanation_stability']['employees']} identical top-3 sets**",
    "direction check": f"**{val['counterfactual_direction_check']['prediction_decreased']}** ({val['counterfactual_direction_check']['share']:.1%})",
    "suppressed share": f"**{internal['suppressed_abs_contribution'].mean() * 100:.1f} percentage points of absolute SHAP per employee**",
    "largest contributor excluded": f"for **{(exp['all_excluded_features']['employees_where_largest_contributor_is_excluded'] / 430):.0%}** of employees",
    "additivity": f"max error {val['shap_max_additivity_error']:.1e}",
}
for k, v in facts.items():
    check(f"README states {k} = {v[:70]}", has(readme, v), "" if has(readme, v) else "NOT FOUND in README")
top = imp.head(11)
drivers = ", ".join(f"{r.label.lower() if i else r.label.lower()} {r.mean_abs_shap_percentage_points:.1f}" for i, r in enumerate(top.itertuples()))
named = pd.concat([internal[f"reason_{i}_feature"] for i in (1, 2, 3)]).value_counts()
for feat, label in (("job_level", "job level"), ("tenure_months", "length of service"), ("last_engagement_score", "engagement"), ("comp_ratio", "pay ratio"), ("work_mode", "work mode")):
    check(f"README most-named count: {label} = {named[feat]}", f"{label} ({named[feat]} employees)" in readme or f"{label} ({named[feat]})" in readme or f"{label} ({named[feat]}" in readme, "")
for feat, rowlabel in (("job_level", "job level"), ("tenure_months", "length of service"), ("last_engagement_score", "engagement score"), ("comp_ratio", "pay vs role mid-point")):
    v = imp.set_index("feature").loc[feat, "mean_abs_shap_percentage_points"]
    check(f"README driver value: {rowlabel} {v:.1f}", f"**{rowlabel} {v:.1f}**" in readme)
check("README total_experience_years importance 3.0 (not named)", f"total experience {imp.set_index('feature').loc['total_experience_years', 'mean_abs_shap_percentage_points']:.1f}" in readme)

# --- generated blocks equal a regeneration and the numbers of checks/tests agree everywhere
n = len(val["validation"])
check("README/audit state the same validation count", f"{n} of {n} output checks pass" in readme and f"{n} of {n} output checks pass" in audit)
check("README/audit state the same test count", f"{tests['passed']} tests passed" in readme and f"{tests['passed']} tests passed" in audit)
check("no stale hand-typed counts remain (25 checks / 45 tests)", not re.search(r"all 25|25 checks|45 passed|45 tests", readme))
check("README total_experience block equals docs/TOTAL_EXPERIENCE_REVIEW.md", (D / "TOTAL_EXPERIENCE_REVIEW.md").read_text(encoding="utf-8").strip() in readme)
for k in ("importance.rank_of_33", "what_if_everyone_at_median_experience.employees_whose_band_changes", "what_if_everyone_at_median_experience.top43_overlap"):
    a, b = k.split("."); val_ = exp[a][b]
    check(f"total_experience review shows {k} = {val_}", str(val_) in readme)
check("README has the three Phase 15 tables", all(h in readme for h in ("### Files inspected", "### Risk-band distribution", "### Ten example employee explanations")))
check("example table has 10 rows with full reasons", readme.count("| EMP-") >= 10 and all(t in readme for t in s10["top_reason_1"]))
check("model hash in README equals recorded Task 13 hash", json.loads((D / "task13_sha256.json").read_text())["files"]["model_artifacts/attrition-v2/model.joblib"][:16] in readme)

bad = [r for r in results if not r[1]]
for name, ok, detail in results:
    print(("PASS " if ok else "FAIL ") + name + (f"   [{detail}]" if (detail and not ok) else ""))
print(f"\n{len(results) - len(bad)} / {len(results)} consistency checks passed")
raise SystemExit(1 if bad else 0)
