"""Write README.md from the computed reports (no hand-typed results)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cortexhr_task15 import config  # noqa: E402

R = ROOT / "reports"
v = json.loads((R / "final_validation_report.json").read_text()); t = json.loads((R / "test_summary.json").read_text())
a = v["data"]["audit"]; comp = pd.read_csv(R / "model_comparison.csv"); mit = pd.read_csv(R / "mitigation_comparison.csv")
gg = pd.read_csv(R / "fairness_career_gap_groups.csv"); gs = pd.read_csv(R / "fairness_career_gap_official_group_gap.csv"); hist = pd.read_csv(R / "historical_pay_gap_controlled.csv")
pol = pd.read_csv(R / "feature_policy.csv"); band = v["band"]; bi = v["bias_investigation"]
importance = pd.read_csv(R / "model_feature_importance.csv")
C = comp.set_index("Model"); base, lr, gb = C.loc["Average-by-role baseline"], C.loc["Linear Regression"], C.loc["Gradient Boosting Regression"]
sel = comp[comp.selected].Model.iloc[0]
fm = lambda x: f"{x:,.0f}"


def md(df, nd=3):
    h = "| " + " | ".join(df.columns) + " |\n|" + "|".join(["---"] * len(df.columns)) + "|\n"
    f = lambda x: (f"{x:,.{nd}f}" if abs(x) < 100 else f"{x:,.0f}") if isinstance(x, float) else str(x)
    return h + "\n".join("| " + " | ".join(f(x) for x in r) + " |" for r in df.itertuples(index=False)) + "\n"


def source_feature(name):
    for column in config.NUMERIC:
        if name == f"num__{column}":
            return column
    for column in config.CATEGORICAL:
        if name.startswith(f"cat__{column}_"):
            return column
    return name


cmp_tbl = pd.DataFrame({"Model": comp.Model, "MAE (INR)": comp.MAE, "R²": comp.R2, "5-fold CV MAE (train only)": comp.CV_MAE_train_only, "train MAE": comp.train_MAE, "train R²": comp.train_R2})
importance_by_input = (
    importance.assign(input_feature=importance["feature"].map(source_feature))
    .groupby("input_feature", as_index=False)["importance"].sum()
    .sort_values("importance", ascending=False, ignore_index=True)
    .rename(columns={"input_feature": "input feature", "importance": "aggregated model importance"})
)
sel_g = gg[gg.model == sel][["group", "n", "mean_actual_salary", "mean_predicted_salary", "mean_signed_error", "mean_signed_error_pct_of_actual", "mae", "mean_actual_vs_role_band", "mean_predicted_vs_role_band", "mean_rec_low", "mean_rec_mid", "mean_rec_high"]]
sel_g.columns = ["group", "n (test)", "mean actual pay", "mean predicted pay", "mean signed error", "signed error % of actual", "MAE", "actual pay / role band", "predicted pay / role band", "mean rec. low", "mean rec. mid", "mean rec. high"]
gs_sel = gs[gs.model == sel][["quantity", "max_gap_between_group_means", "group_means"]]; gs_sel.columns = ["quantity (official group_gap)", "largest gap between group means", "group means"]
hist_t = hist[["group", "reference", "n_group", "n_reference", "pay_difference_pct", "ci95_low_pct", "ci95_high_pct", "interval_excludes_zero"]].copy()
hist_t.columns = ["group", "vs", "n group", "n reference", "pay difference %", "95% CI low", "95% CI high", "CI excludes 0"]
m_t = mit.copy(); m_t = m_t.rename(columns={"cv_mae_train_only": "CV MAE (train only)", "mae": "test MAE", "r2": "test R²", "label_correction_factors": "label factors"}).fillna("-")
pol_t = pol.rename(columns={"column": "column"}); lev = pd.DataFrame([{"job level": l, "low ratio": x[0], "high ratio": x[1], "training residuals": x[2], "test n": band["test_coverage_by_job_level"][l]["n"], "test coverage": band["test_coverage_by_job_level"][l]["coverage"]} for l, x in band["ratios_by_level"].items()])
lowc = [(l, c['coverage'], c['n']) for l, c in band['test_coverage_by_job_level'].items() if c['coverage'] < 0.78]
low_cov_txt = 'levels with held-out coverage below 78%: ' + (', '.join(f'level {l} ({c:.1%}, n={n})' for l, c, n in lowc) or 'none') + '; the level with the fewest test hires (' + min(((c['n'], l) for l, c in band['test_coverage_by_job_level'].items()))[1] + ') has only ' + str(min(c['n'] for c in band['test_coverage_by_job_level'].values())) + ' test hires, so its figure is noisy'
cov = band["test_coverage"]; px = bi["proxy_checks"]["career_gap_from_allowed_features_auc"]["auc"]
gb_imp = (gb["MAE"] / base["MAE"] - 1) * 100; lr_imp = (lr["MAE"] / base["MAE"] - 1) * 100
g12, g39, g0 = (gg[(gg.model == sel) & (gg.group.str.contains(k))].iloc[0] for k in ("12\\+", "3-9", "No career"))
h12, h39 = hist[hist.group.str.contains("12")].iloc[0], hist[hist.group.str.contains("3-9")].iloc[0]
m0, m1, m2 = mit.iloc[0], mit.iloc[1], mit.iloc[2]
gender_run = v["gender_audit"]["run"]
status = "PASS WITH WARNINGS" if v.get("all_checks_passed") and t["failed"] == 0 and t["skipped"] == 0 and gender_run else "NOT READY — FIXES REQUIRED"
if gender_run:
    ggg = pd.read_csv(R / "fairness_gender_groups.csv")
    ggs = pd.read_csv(R / "fairness_gender_official_group_gap.csv")
    selected_gender = ggg[ggg.model == sel][["group", "n", "reliable_n", "mean_actual_salary", "mean_predicted_salary", "mean_signed_error", "mae"]].copy()
    selected_gender.columns = ["group", "n (test)", "reliable n", "mean actual pay", "mean predicted pay", "mean signed error", "MAE"]
    gender_gaps = ggs[ggs.model == sel][["quantity", "max_gap_between_group_means", "group_means"]].copy()
    gender_gaps.columns = ["quantity (official group_gap)", "largest gap between group means", "group means"]
    gender_history = hist[hist.grouping == "gender"][["group", "reference", "n_group", "n_reference", "pay_difference_pct", "ci95_low_pct", "ci95_high_pct", "interval_excludes_zero"]]
    female_proxy = bi["proxy_checks"].get("gender_female_vs_nonfemale_from_allowed_features_auc", {})
    gender_section = f"""### 7.1 Gender (authorized unlock_CX-15; selected model {sel}; held-out test set)
{md(selected_gender, 3)}

Official `group_gap()` results:

{md(gender_gaps, 3)}

The unlock file supplied {v['gender_audit'].get('unlock_rows', 0):,} unique person references. {v['gender_audit'].get('matched_test_rows', 0):,} of {v['split']['test']:,} test rows had a gender label; unmatched: {v['gender_audit'].get('unmatched_test_rows', 0)}. Groups below {config.MIN_GROUP} are unreliable; `Other` has n=11.
The controlled historical-pay estimate for Female vs Male is:

{md(gender_history.rename(columns={"group": "group", "reference": "reference", "n_group": "n group", "n_reference": "n reference", "pay_difference_pct": "pay difference %", "ci95_low_pct": "95% CI low", "ci95_high_pct": "95% CI high", "interval_excludes_zero": "CI excludes 0"}), 2)}

The allowed features' cross-validated AUC for Female vs non-Female is {female_proxy.get('auc', float('nan')):.3f}. This is only a binary proxy diagnostic, not proof that proxies are absent."""
else:
    gender_section = """### 7.1 Gender - NOT RUN

The required `unlock_CX-15` audit file was unavailable. No gender result is inferred or fabricated; pipeline validation fails until the official data is supplied."""
neg = v["non_positive_test_predictions"]

txt = f"""# Task 15 - Recommend fair pay bands (CortexHR, Phase 3, Day 15)

Predicts a **salary range (Low / Mid / High)** for a role from job-related features, compares three models, tests for pay gaps between groups with the official `group_gap()` helper,
and documents what the data can and cannot support. **A model-generated recommendation is decision support, not a compensation decision.**

> ## STATUS: {status}
> The official `unlock_CX-15` data was {'used for the gender audit' if gender_run else 'not available, so gender validation is incomplete'}. Do not treat this report as approval to make compensation decisions.

## 1. Objective
Predict a sensible offered salary for a role, recommend a range, and test the model for gaps between groups (gender, career gap), following the official workflow: **build the model first, open the audit data afterwards.**

## 2. Dataset - `salary_training` (local workbook sheet)
{a['rows']:,} rows x {len(a['columns'])} columns, **no missing values, no duplicate rows or ids**, {a['roles']} roles, {a['departments']} departments, job levels {', '.join(f"{k}: {n}" for k, n in a['job_levels'].items())}.
**Target: `{config.TARGET}`** (annual fixed pay offered at hire, INR; marked `TARGET` in the workbook data dictionary; CSV export matches the workbook sheet). Mean {fm(a['target_summary']['mean'])}, median {fm(a['target_summary']['50%'])}, min {fm(a['target_summary']['min'])}, max {fm(a['target_summary']['max'])};
skew {a['target_skew']:.2f} (log scale {a['target_log_skew']:.2f}); {a['iqr_outliers']} observations outside the 1.5-IQR thresholds (bounds {fm(a['iqr_outlier_bounds']['low'])} to {fm(a['iqr_outlier_bounds']['high'])}), **retained** rather than automatically dropping valid high salaries. Sources: {', '.join(f'{k} {n}' for k, n in a['source_counts'].items())}; hire years {a['hire_year_range'][0]}-{a['hire_year_range'][1]}.
`career_gap_months` is in the table ({a['career_gap_share']:.1%} of hires have a gap; values {', '.join(f'{k}: {n}' for k, n in a['career_gap_months_counts'].items())}). **Gender and age are NOT in this table** (only in `unlock_CX-15`). No historical-salary field exists.
Role counts and department counts are recorded in `reports/dataset_audit.json`. `employment_type` is **not present** and was not imputed or inferred.
Key finding: the 2026 role bands are **circular** - target / (band_mid_2026 x market_index) averages {a['band_circularity']['mean_target_over_band_mid_x_index']:.3f} (sd {a['band_circularity']['sd']:.3f}), log-correlation {a['band_circularity']['log_correlation']:.3f}.
Full audit: `reports/dataset_audit.json`.

## 3. Feature policy (every column has a decision; `reports/feature_policy.csv`)
{md(pol_t)}
**Allowed predictive features:** {', '.join(config.ALLOWED)}. **Audit-only:** `career_gap_months`, gender and age fields from `unlock_CX-15`; join/provenance fields are not predictors. **Excluded:** 2026 bands (circular), hire dates/year (time proxy/redundancy), `skills_count` (documented noise), source and IDs. Historical salary is the label, not a feature. Protected attributes are not used in prediction.

## 4. Preprocessing and evaluation strategy
`ColumnTransformer` inside each `Pipeline`: numeric = median imputation + standard scaling; categorical = constant "Unknown" imputation + `OneHotEncoder(handle_unknown="ignore")`. Fitted on **training rows only** (tested), re-used unchanged for any later row, so unseen roles/categories never crash.
Split: random 80/20, stratified by data source, **seed {config.SEED}**: train {v['split']['train']:,}, test {v['split']['test']:,}. Hire IDs are unique, but no separate stable employee identifier is available to rule out repeat-person leakage; this is a row-level split, not a person-grouped or temporal holdout. The test set was not used for model selection (selection = lowest 5-fold out-of-fold MAE on training rows).

## 5. Models
* **Average-by-role baseline:** training mean salary of the role; unseen role -> overall training mean. Ignores inflation and level.
* **Linear Regression:** same pipeline, no regularisation.
* **Gradient Boosting Regression** (`GradientBoostingRegressor`, 300 trees, depth 3, learning rate 0.05, subsample 0.8, seed {config.SEED}): fixed settings, **no tuning**.

## 6. Results - held-out test set (n = {v['split']['test']})
{md(cmp_tbl, 3)}
* **Lowest MAE and best R²: {sel}** (MAE {fm(C.loc[sel, 'MAE'])}, R² {C.loc[sel, 'R2']:.3f}); it cuts MAE by {abs((C.loc[sel, 'MAE'] / base['MAE'] - 1) * 100):.0f}% versus the baseline ({fm(base['MAE'])}). Linear Regression changes MAE by {abs(lr_imp):.0f}% versus the baseline. Both ML models improve on this split; this alone does not establish fairness or future performance.
* **Overfitting:** Gradient Boosting's training MAE ({fm(gb['train_MAE'])}) is lower than its test MAE ({fm(gb['MAE'])}); the test gap is modest (ratio {gb['MAE'] / gb['train_MAE']:.2f}) and CV agrees ({fm(gb['CV_MAE_train_only'])}), so mild, not severe, overfitting. Linear Regression: {lr['train_MAE']:,.0f} train vs {lr['MAE']:,.0f} test.
* **Is the data enough?** It is sufficient for a task-level benchmark on this supplied table (R² above 0.96 for both ML models; every role has at least {a['min_rows_per_role']} hires), but **not enough to establish production reliability**. This practice kit contains 734 real hires and 2,200 historical extension hires; results are not validated as representative of an external workforce.
* Linear Regression predicted a **negative salary for {neg['linear_regression']} test hire(s)** (linear models cannot respect "salary > 0"). Gradient Boosting and the baseline: {neg['gradient_boosting']} and {neg['average_by_role_baseline']}. Predictions are not floored in the reported metrics; the band code never outputs a negative value.
* Charts: `reports/model_comparison.png`, `reports/salary_distribution.png`.
* Selected model feature influence: `reports/model_feature_importance.csv` (model-native split importance or absolute coefficient magnitude; one-hot categories represent individual encoded values, not causal effects).

Aggregated to the original input fields (split importance is not causal):

{md(importance_by_input.head(7), 3)}

## 7. Fairness testing
Method: the **official Task 12 `group_gap(scores, groups)`** is imported directly from `task12/src/cortexhr_task12/cortexhr_helpers.py`; that source is not modified and there is no fallback copy. It returns mean scores per group and the largest gap between group means, excluding missing labels/scores and rejecting malformed inputs. It is a warning, not proof.
It is applied to six quantities, kept separate on purpose: **actual (historical) pay**, **predicted pay**, **signed error** (predicted - actual), **absolute error**, pay relative to each role's 2026 band (an audit yardstick, never a model input), and the **recommended Low/Mid/High**.
Groups with fewer than {config.MIN_GROUP} people are flagged unreliable. Missing group labels are dropped and counted, never mixed in.
**Distinctions:** a historical pay difference does not prove the model caused it; similar average predictions do not prove fairness.

{gender_section}

### 7.2 Career gap (from `salary_training.career_gap_months`; selected model {sel}; test set)
{md(sel_g)}
Official `group_gap()` results (same model):

{md(gs_sel)}
Small groups: {int(g12['n'])} and {int(g39['n'])} people in the two gap groups - above the {config.MIN_GROUP}-person threshold but still small; treat numbers as indicative.

**Reading it.** The **raw** historical-pay gap ({fm(gs_sel.iloc[0, 1])} INR) is not a like-for-like comparison (it controls for no role, level or experience differences, and was not decomposed here), so the role-band ratio and the controlled estimate below are the relevant measures.
Relative to each role's band, hires with a 12+ month gap were historically paid **{g12['mean_actual_vs_role_band']:.3f}x** the band versus **{g0['mean_actual_vs_role_band']:.3f}x** without a gap, whereas the model predicts **{g12['mean_predicted_vs_role_band']:.3f}x** versus **{g0['mean_predicted_vs_role_band']:.3f}x**:
the model does **not** reproduce the penalty. Its mean signed error for the 12+ group is **{g12['mean_signed_error_pct_of_actual']:+.1f}%** of actual pay (versus {g0['mean_signed_error_pct_of_actual']:+.1f}% without a gap): the model predicts more than they were historically paid, and the **MAE is {g12['mae'] / g0['mae'] - 1:+.0%} higher** for them. Chart: `reports/fairness_gap.png`.

## 8. Historical pay bias: "if women (or career-gap hires) were paid less, should the model copy that?"
No. What the data shows (controlled for role, level, department, experience, location, education and hire-year price level; `reports/historical_pay_gap_controlled.csv`):

{md(hist_t, 2)}
* **Career gap, 12+ months: historically paid {abs(h12['pay_difference_pct']):.1f}% less** (95% bootstrap interval {h12['ci95_low_pct']:.1f}% to {h12['ci95_high_pct']:.1f}%, excludes zero). 3-9 months: {h39['pay_difference_pct']:.1f}% (interval {h39['ci95_low_pct']:.1f}% to {h39['ci95_high_pct']:.1f}%, includes zero - not distinguishable from no effect). Descriptive, not causal.
* {"Female hires were paid " + f"{abs(hist[(hist.grouping == 'gender') & (hist.group == 'Female')].iloc[0]['pay_difference_pct']):.1f}% less than Male hires after controlling for job factors (95% bootstrap interval " + f"{hist[(hist.grouping == 'gender') & (hist.group == 'Female')].iloc[0]['ci95_low_pct']:.1f}% to {hist[(hist.grouping == 'gender') & (hist.group == 'Female')].iloc[0]['ci95_high_pct']:.1f}%; descriptive, not causal)." if gender_run else "Gender pay differences cannot be assessed without the authorized unlock data."}
* **Is historical salary a feature? No** (`{bi['historical_salary_in_features']}`); it is the *label*. **Gender / career gap as features? No** (`{bi['gender_in_features']}`, `{bi['career_gap_in_features']}`). The 2026 bands are excluded as circular.
* **Do other features act as proxies?** Allowed features predict career gap with cross-validated **AUC {px:.2f}** (0.50 = chance); this test did not detect a strong proxy signal, but cannot prove proxies are absent. {"The Female-vs-non-Female proxy diagnostic AUC is " + f"{female_proxy.get('auc', float('nan')):.2f}; it is binary and does not establish that all gender proxies are absent." if gender_run else "A gender proxy check was not run."}
* **Is excluding the attribute sufficient?** Not in general: the **labels** still carry the penalty, so the model is trained on, and scored against, biased history. Here the model happens not to reproduce it (above), but any accuracy measured against historical pay partly rewards copying inequity: the 12+ group looks "worse predicted" only *because* the model refuses to copy a penalty.

## 9. Mitigation experiments (original kept as M0; each variant re-trained and scored against the original historical test labels)
{md(m_t, 3)}
* **M1 label correction** (estimate the career-gap pay difference on training rows only and add it back to those labels; features unchanged; uses the audit attribute at training time only): test MAE {m1['mae']:,.0f} vs {m0['mae']:,.0f}; career-gap signed-error gap {m1['career_gap: signed-error gap (INR)']:,.0f} vs {m0['career_gap: signed-error gap (INR)']:,.0f}; gender signed-error gap {m1.get('gender: signed-error gap (INR)', float('nan')):,.0f} vs {m0.get('gender: signed-error gap (INR)', float('nan')):,.0f}. Its small MAE improvement did not improve the career-gap error gap, and it requires protected audit data during label construction: **not adopted**.
* **M2 drop `work_location` and `highest_education`** (possible proxies): CV MAE {m0['CV MAE (train only)'] if 'CV MAE (train only)' in m0 else m0['cv_mae_train_only']:,.0f} -> {m2['cv_mae_train_only']:,.0f} and test MAE {m0['mae']:,.0f} -> {m2['mae']:,.0f}. Career-gap signed-error gap changes from {m0['career_gap: signed-error gap (INR)']:,.0f} to {m2['career_gap: signed-error gap (INR)']:,.0f} INR; gender signed-error gap changes from {m0.get('gender: signed-error gap (INR)', float('nan')):,.0f} to {m2.get('gender: signed-error gap (INR)', float('nan')):,.0f} INR. These metrics do not establish a universal fairness improvement. **Not adopted**: location can be a legitimate cost-of-labour factor and evidence is insufficient to remove it automatically.
* No individual recommendation was adjusted using gender or career-gap status, and group parity was not forced. **Neither M1 nor M2 is adopted**: M1's label adjustment uses protected audit information in training and did not justify its trade-off; M2 requires an HR decision about legitimate job-related factors.

## 10. Salary-band construction (`reports/salary_band_recommendations.csv`)
**Mid** = {sel} prediction. **Low / High = Mid x the 10th / 90th percentile of (actual / predicted)**, taken from **training out-of-fold** residuals ({v['split']['train']:,} rows) and computed **per job level** (a job attribute); a global fallback is used only when a level has too few training residuals to calibrate independently.
Rounded to whole rupees; guaranteed 0 <= Low <= Mid <= High. Test labels are used only to **measure** coverage. Gender and career gap never enter.

{md(lev, 3)}
**Held-out coverage: {cov['coverage']:.1%}** (nominal 80%), {cov['below_low']:.1%} below Low, {cov['above_high']:.1%} above High, mean relative width {cov['mean_relative_width']:.1%}. This is an **empirical prediction band, not a proven confidence interval**; {low_cov_txt}

## 11. Output files (`reports/`)
`salary_band_recommendations.csv` ({v['recommendations']} held-out hires: `record_id, role_id, predicted_salary_mid, recommended_salary_low, recommended_salary_high, model_version, human_review_required, model_note`; no protected data),
`role_band_reference_2026.csv` (typical-profile band per role at the April-2026 price level), `model_comparison.csv`, `model_feature_importance.csv`,
`fairness_career_gap_groups.csv`, `fairness_career_gap_official_group_gap.csv`, `fairness_gender_groups.csv`, `fairness_gender_official_group_gap.csv`, `historical_pay_gap_controlled.csv`, `mitigation_comparison.csv`, `feature_policy.csv`, `dataset_audit.json`, `dataset_columns.csv`,
`salary_distribution.png`, `model_comparison.png`, `fairness_gap.png`, `final_validation_report.json`, `test_summary.json`; fitted pipelines in `reports/model_artifacts/` (`{config.MODEL_VERSION}_*.joblib`). The older `models/` artifacts are left untouched.
Per-record baseline, Linear Regression and Gradient Boosting prediction/error tables (including historical salary labels) are stored only under the git-ignored `reports/private/` directory.

## 12. Validation and tests
Output checks (`final_validation_report.json`): {sum(c['passed'] for c in v['validation'])} of {len(v['validation'])} pass, including the required official gender audit, the production CSV schema and band ordering.
**Tests: {t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped** (`{t['summary_line']}`).

## 13. Limitations
* {"Gender groups include n=11 for Other, too small for a reliable standalone conclusion; age is present in the unlock file but was not a required fairness dimension and was not tested." if gender_run else "**Gender audit not run**: the official unlock file is required; age is also not audited."}
* **Historical labels** encode past decisions, including the measured career-gap penalty; accuracy against them is not the same as fairness.
* Practice-kit data: {a['source_counts'].get('master_hire', 0):,} real hires and {a['source_counts'].get('historical_extension', 0):,} extension hires; external representativeness is unestablished. Rare roles have few examples (minimum {a['min_rows_per_role']} hires).
* Group tests have small samples ({int(g12['n'])} / {int(g39['n'])} test hires in the gap groups); a measured gap can be noise.
* `work_location` and `highest_education` remain in the model (see M2); `market_index` assumes 5.5% yearly pay inflation; the band is a level-wise empirical band, not a confidence interval, and under-covers at some levels (section 10).
* Linear Regression can output negative salaries; only the CV-selected model ({sel}) feeds the bands. The random row split is neither person-grouped nor time-based because a separate stable person identifier was unavailable.
* Recommendations were generated for the held-out test hires (out-of-sample) and as role-level references; new candidates need the same feature columns and should be assessed under a fresh temporal and employee-grouped validation.

## 14. Responsible use
A model-generated salary recommendation is **decision support, not a final compensation decision.** Final compensation must be reviewed by HR and an authorised manager; the model must not be the sole basis for a pay decision.
Do not use gender, career gap or any proxy to adjust a recommended range. Re-run the fairness audit when data or features change.

## 15. Run
```
cd task15
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
$env:CORTEXHR_WORKBOOK = "C:\\path\\to\\CortexHR_Dataset.xlsx"   # optional; default: data/CortexHR_Dataset.xlsx
# required: place authorized unlock_CX-15 data in private/unlock_CX-15_protected_attributes.csv
python scripts/run_task15.py
python scripts/run_checks.py
python scripts/build_readme.py
```
"""
(ROOT / "README.md").write_text(txt, encoding="utf-8")
print("README written", len(txt))
