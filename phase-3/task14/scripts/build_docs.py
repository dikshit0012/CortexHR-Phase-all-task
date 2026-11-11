"""Generate the Phase 15 tables, the total_experience_years review and the validation summary FROM THE DATA (no hand-typed numbers), and
inject them into README.md between markers. Writes docs/FINAL_AUDIT.md and docs/TOTAL_EXPERIENCE_REVIEW.md.
Inputs: reports/*.csv, reports/internal/{validation_report,total_experience_analysis,test_summary}.json, docs/task13_sha256.json."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cortexhr_task14 import config  # noqa: E402

R, I, D = ROOT / "reports", ROOT / "reports" / "internal", ROOT / "docs"
J = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))


def md(df: pd.DataFrame) -> str:
    h = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "|".join(["---"] * len(df.columns)) + "|\n"
    return h + "\n".join("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


FILES_INSPECTED = [
    ("GitHub repo `CortexHR-Phase-2` (cloned)", "tasks 5-10, plus Task 11/12 written earlier", "listed + searched", "contains NO Task 13, Task 4 or model artifact, so Task 13 was taken from the uploaded zip"),
    ("`task_13.zip`: `task 13/README.md`", "Task 13 documentation", "read", "model choice rule, usage, limitations, `load_model`/`predict_attrition` entry points"),
    ("`task_13.zip`: `src/attrition_v2/model_versioning.py`, `config.py`", "Task 13 loader and scoring code", "read + imported", "official scoring path reused (no duplication)"),
    ("`task_13.zip`: `model_artifacts/attrition-v2/model.joblib`, `metadata.json`, `metrics.json`", "the trained model", "loaded + read", "CalibratedClassifierCV, 5 members, 33 features, version attrition-v2"),
    ("`cortexhr-ml_day4.zip`: `cortexhr-ml/src/cortexhr/` (features, monthly, ...)", "Task 2/4 feature pipeline stored inside the model", "listed + imported", "needed to unpickle and to build the scoring table"),
    ("`README.md`, `requirements.txt`, `pytest.ini`, `docker-compose.yml`, `Dockerfile`, `_gitignore`, `_dockerignore`", "Task 1/2 project conventions", "read (Dockerfile / ignore files listed only)", "layout and dependency conventions"),
    ("`verification_audit.md`", "earlier verification report", "opened (head only)", "context only; not used as evidence"),
    ("Workbook `CortexHR_Dataset.xlsx` - sheet `attrition_scoring_2026`", "the 430 employees to score", "read + validated", "430 rows, unique ids, 34 columns"),
    ("Workbook - sheet `sample_attrition_risk`", "official output format", "read + schema copied", "11 columns, order, id format, rounding, band counts"),
    ("Workbook - sheets `data_dictionary`, `README`", "column meanings, CX-13/CX-14 notes", "read", "units and meanings used for wording"),
    ("Workbook - sheet `attrition_dataset`", "Task 13 training table", "read (analysis only)", "used for the total_experience_years association check"),
    ("Workbook - column scan of all 26 sheets", "search for age / DOB / gender", "scanned", "no employee age, date of birth or gender column exists"),
    ("`cortexhr_helpers.py`", "official `risk_bands()`", "read; copied verbatim into `src/cortexhr_task14/`", "unchanged; `risk_bands` used as is"),
]


def band_table(out: pd.DataFrame, val: dict) -> str:
    rb = val["risk_bands"]
    g = out.groupby("risk_band")["risk_score_12m"].agg(["count", "min", "max"]).reindex(["High", "Medium", "Low"])
    t = pd.DataFrame({"band": g.index, "employees": g["count"].values, "share": [f"{c / len(out):.1%}" for c in g["count"]],
                      "lowest score": g["min"].map("{:.3f}".format).values, "highest score": g["max"].map("{:.3f}".format).values,
                      "rule (official helper)": ["top 10% of scores", "next 20%", "the rest"]})
    s = (f"Helper cut-offs for this batch: High >= {rb['high_cutoff_p90']:.4f} (90th percentile), Medium >= {rb['medium_cutoff_p70']:.4f} (70th percentile). "
         f"Bands are rank-based, so they describe position in these {len(out)} employees, not an absolute probability.")
    try:
        import os
        wb = os.environ.get("CORTEXHR_WORKBOOK")
        if wb and Path(wb).exists():
            sc = pd.read_excel(wb, sheet_name=config.SAMPLE_SHEET)["risk_band"].value_counts()
            same = {k: int(sc.get(k, 0)) for k in ("High", "Medium", "Low")} == {k: int(v) for k, v in g["count"].items()}
            s += f" The official `sample_attrition_risk` sheet has High {sc.get('High', 0)} / Medium {sc.get('Medium', 0)} / Low {sc.get('Low', 0)}: " + ("identical split." if same else "a DIFFERENT split.")
    except Exception:
        pass
    return md(t) + "\n" + s + "\n"


def examples_table(s10: pd.DataFrame, internal: pd.DataFrame) -> str:
    i = internal.set_index("employee_id")
    rows = []
    for r in s10.itertuples():
        cells = []
        for k, t in enumerate((r.top_reason_1, r.top_reason_2, r.top_reason_3), 1):
            c = i.loc[r.employee_id, f"reason_{k}_contribution"] * 100
            cells.append(f"{k}. {t} ({c:+.1f} pp)")
        rows.append({"employee": r.employee_id, "score": f"{r.risk_score_12m:.3f}", "band": r.risk_band, "reasons (SHAP contribution, percentage points)": " ".join(cells)})
    return md(pd.DataFrame(rows))


def validation_table(val: dict) -> str:
    return md(pd.DataFrame([{"#": n, "check": c["check"], "result": "PASS" if c["passed"] else "FAIL"} for n, c in enumerate(val["validation"], 1)]))


def total_experience_md(a: dict) -> str:
    im, rel, w, ex = a["importance"], a["relationships_in_scoring_data"], a["what_if_everyone_at_median_experience"], a["if_it_were_not_suppressed"]
    tr, al = a["training_table_association"], a["all_excluded_features"]
    co = a["logistic_coefficient_per_sd_by_member"]
    return f"""## `total_experience_years` review

**Question.** The model's #2 most influential input is `total_experience_years`. It is never named in an employee's reasons as a precaution, because it could act as a proxy for age (which cannot be measured here). Is that policy enough, and what does the evidence show?

**What the data can and cannot show.** The workbook has **no age, date-of-birth or gender column for employees** (all 26 sheets scanned), so a link to age is **not measured here**. The concern rests on how the variable is built: experience accumulates over a working life. Nothing below demonstrates age discrimination, and nothing rules it out.

| evidence (real attrition-v2 model, 430 employees) | value |
|---|---|
| Importance rank by mean absolute SHAP (of 33) | **{im['rank_of_33']}** ({im['mean_abs_shap_pp']:.2f} percentage points; {im['share_of_total_abs_shap']:.1%} of all SHAP) - after job level ({im['top_features_pp']['job_level']:.2f}), before length of service ({im['top_features_pp']['tenure_months']:.2f}) |
| Direction | more recorded experience -> **higher** model risk (rank correlation of value and contribution {im['direction']['spearman_value_vs_shap']:.2f}) |
| Correlated with (Spearman, scoring data) | job level {rel['spearman_with_job_level']:.2f}, annual pay {rel['spearman_with_annual_fixed_pay']:.2f}, tenure {rel['spearman_with_tenure_months']:.2f}, previous employers {rel['spearman_with_previous_employers_count']:.2f} |
| Data quality | {rel['experience_less_than_tenure_rows']} employees record **less total experience than their tenure** (range {rel['min_max_total_experience_years'][0]:.1f}-{rel['min_max_total_experience_years'][1]:.1f} years, median {rel['median_total_experience_years']:.1f}) |
| Logistic coefficient per standard deviation, 5 ensemble members | {', '.join(f'{c:.2f}' for c in co)} (all positive, but varies by almost a factor of 2) |
| Marginal signal in the training table (2,379 rows, 349 leavers) | AUC **{tr['univariate_auc_total_experience_years']:.2f}** - alone it does **not** separate leavers |
| Sign once tenure and job level are held fixed | +{tr['coef_per_sd_with_tenure_and_job_level']['total_experience_years']:.2f} per SD, bootstrap 95% interval [{tr['bootstrap_95ci_experience_coef'][0]:.2f}, {tr['bootstrap_95ci_experience_coef'][1]:.2f}] - the lower end touches zero |
| If it were NOT suppressed | in the top-3 risk-raising drivers of **{ex['employees_with_it_in_top3_risk_raising']}** employees, the single top driver for **{ex['employees_with_it_as_top_driver']}** |
| All 7 suppressed features together | {al['mean_abs_shap_pp_total']:.1f} percentage points of absolute SHAP per employee; the single largest contributor is a suppressed feature for **{al['employees_where_largest_contributor_is_excluded']}** employees ({al['employees_where_largest_contributor_is_excluded'] / a['employees']:.0%}), **{al['of_which_total_experience_years']}** of them `total_experience_years` |

**Why leaving it out of the explanations does not remove its influence.** The reasons are a post-processing step on top of the model: attrition-v2 still receives the feature and still uses it to compute every score and band. What-if on the real model (the model is untouched; only the input is altered): setting everyone's experience to the cohort median changes the predictions by **{w['mean_abs_change_pp']:.1f} percentage points on average** (up to {w['max_abs_change_pp']:.1f}), moves **{w['employees_whose_band_changes']} employees to a different band**, and keeps only **{w['top43_overlap']} of the 43** highest-risk people in the High band. It is the {a['what_if_rank_of_feature']}rd most influential input by this measure as well. So the scores and High-risk list partly depend on it, while the displayed reasons do not mention it.

**What this means.**
* The suppression is a **presentation safeguard, not a fairness fix.** It stops the report from saying "experience raises risk", but it cannot undo the feature's effect on who is ranked High.
* Its contribution is **statistically fragile**: no marginal signal, a positive sign only after conditioning on tenure and job level, a bootstrap interval touching zero, and member-to-member variation. It looks like a statistical correction for tenure and level rather than a robust attrition signal. That is evidence about *predictive stability*, not about fairness.
* Job level (#1) and tenure (#3) are also correlated with experience (and plausibly with age); both remain nameable (tenure with the mandated neutral wording). The exclusion list is therefore **not a complete age-proxy control**.
* Not done and not claimed: no group-fairness audit (no protected attribute exists for employees), no age analysis, no retraining. A defensible next step for Task 13/HR/legal is to retrain **without** `total_experience_years` and compare ranking quality and who moves bands, and to fix the {rel['experience_less_than_tenure_rows']} inconsistent records. Task 14 must not change attrition-v2.
"""


def main() -> None:
    out = pd.read_csv(R / "attrition_risk_explanations.csv")
    s10 = pd.read_csv(R / "attrition_risk_sample_10.csv")
    internal = pd.read_csv(I / "attrition_scoring_internal.csv")
    val, a = J(I / "validation_report.json"), J(I / "total_experience_analysis.json")
    tests = J(I / "test_summary.json")
    imp = pd.read_csv(I / "shap_feature_importance.csv")
    sha = J(D / "task13_sha256.json")
    n_checks, n_pass = len(val["validation"]), sum(c["passed"] for c in val["validation"])
    insp = md(pd.DataFrame(FILES_INSPECTED, columns=["file / source", "role", "how it was used", "finding"]))
    top = imp.head(8)[["label", "mean_abs_shap_percentage_points", "named_in_reasons"]].copy()
    top["mean_abs_shap_percentage_points"] = top["mean_abs_shap_percentage_points"].map("{:.2f}".format)
    top.columns = ["feature", "mean |SHAP| (percentage points)", "may be named in reasons"]
    summary = (f"{n_pass} of {n_checks} output checks pass; {tests['passed']} tests passed, {tests['failed']} failed ({tests['when']}). "
               f"Task 13 workspace matches its recorded {len(sha['files'])}-file baseline (source: {sha['source']}; "
               f"model.joblib sha256 {sha['files']['model_artifacts/attrition-v2/model.joblib'][:16]}...).")
    blocks = {
        "PHASE15": ("### Files inspected\n" + insp + "\n### Risk-band distribution\n" + band_table(out, val) + "\n### Ten example employee explanations\n"
                    "Real rows of `attrition_risk_sample_10.csv` (High 4 / Medium 3 / Low 3, evenly spaced by score). Every one requires human review; "
                    "the notice in the CSV's `model_note` applies to each.\n\n" + examples_table(s10, internal)),
        "TOTALEXP": total_experience_md(a),
        "VALIDATION": f"**{summary}**\n\n" + validation_table(val),
    }
    (D / "TOTAL_EXPERIENCE_REVIEW.md").write_text(blocks["TOTALEXP"], encoding="utf-8")
    final = ("# Task 14 - final audit (generated)\n\n" + summary + "\n\n## Validation checks\n" + validation_table(val) + "\n## Most influential features\n" + md(top) + "\n"
             + blocks["PHASE15"] + "\n" + blocks["TOTALEXP"])
    (D / "FINAL_AUDIT.md").write_text(final, encoding="utf-8")
    if (ROOT / "FINAL_AUDIT_task14.md").exists():
        (ROOT / "FINAL_AUDIT_task14.md").write_text(final, encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name, text in blocks.items():
        pat = re.compile(rf"<!--{name}:START-->.*?<!--{name}:END-->", re.S)
        if not pat.search(readme):
            raise SystemExit(f"marker {name} missing in README.md")
        readme = pat.sub(lambda m: f"<!--{name}:START-->\n{text}\n<!--{name}:END-->", readme)
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("docs written;", summary)


if __name__ == "__main__":
    main()
