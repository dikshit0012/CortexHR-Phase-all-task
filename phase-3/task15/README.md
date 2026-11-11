# Task 15 - Recommend fair pay bands (CortexHR, Phase 3, Day 15)

Predicts a **salary range (Low / Mid / High)** for a role from job-related features, compares three models, tests for pay gaps between groups with the official `group_gap()` helper,
and documents what the data can and cannot support. **A model-generated recommendation is decision support, not a compensation decision.**

> ## STATUS: PASS WITH WARNINGS
> The official `unlock_CX-15` data was used for the gender audit. Do not treat this report as approval to make compensation decisions.

## 1. Objective
Predict a sensible offered salary for a role, recommend a range, and test the model for gaps between groups (gender, career gap), following the official workflow: **build the model first, open the audit data afterwards.**

## 2. Dataset - `salary_training` (local workbook sheet)
2,934 rows x 17 columns, **no missing values, no duplicate rows or ids**, 47 roles, 10 departments, job levels 1: 794, 2: 997, 3: 694, 4: 348, 5: 101.
**Target: `offered_annual_fixed`** (annual fixed pay offered at hire, INR; marked `TARGET` in the workbook data dictionary; CSV export matches the workbook sheet). Mean 834,428, median 624,192, min 114,432, max 4,197,972;
skew 1.76 (log scale 0.28); 134 observations outside the 1.5-IQR thresholds (bounds -720,096 to 2,214,936), **retained** rather than automatically dropping valid high salaries. Sources: historical_extension 2200, master_hire 734; hire years 2012-2026.
`career_gap_months` is in the table (13.1% of hires have a gap; values 0: 2549, 3: 71, 6: 53, 9: 62, 12: 60, 18: 53, 24: 41, 36: 45). **Gender and age are NOT in this table** (only in `unlock_CX-15`). No historical-salary field exists.
Role counts and department counts are recorded in `reports/dataset_audit.json`. `employment_type` is **not present** and was not imputed or inferred.
Key finding: the 2026 role bands are **circular** - target / (band_mid_2026 x market_index) averages 0.992 (sd 0.081), log-correlation 0.993.
Full audit: `reports/dataset_audit.json`.

## 3. Feature policy (every column has a decision; `reports/feature_policy.csv`)
| column | decision | reason |
|---|---|---|
| role_id | allowed | the job itself: the main driver of the value of a role |
| job_level | allowed | responsibility level (1 Associate ... 7 CEO) |
| department_id | allowed | function / pay market of the role |
| prior_experience_years | allowed | relevant experience before joining; a standard pay factor |
| market_index | allowed | pay-inflation index (5.5% a year): converts hire-year pay to a common price level; not a person attribute |
| work_location | allowed (governed) | cost-of-labour category; may proxy for region, so it is evaluated and can be removed (mitigation M2) |
| highest_education | allowed (governed) | qualification; may proxy for socio-economic background, so it is evaluated and can be removed (mitigation M2) |
| career_gap_months | audit-only | required for fairness testing; using it would build a past pay penalty into recommendations |
| gender | audit-only | protected attribute from unlock_CX-15; fairness analysis only, never a predictive input |
| age_years | audit-only | protected attribute from unlock_CX-15; fairness analysis only, never a predictive input |
| age_band | audit-only | protected attribute from unlock_CX-15; fairness analysis only, never a predictive input |
| person_ref | audit-only | unlock_CX-15 join key; used only to link authorized audit labels and never a predictive input |
| record_type | audit-only | unlock_CX-15 provenance metadata; never a predictive input |
| band_min_2026 | excluded | 2026 role band: circular (target / (band_mid x market_index) averages 0.99, log-correlation 0.99); it is the pay structure being recommended, and role_id already carries the role information |
| band_mid_2026 | excluded | as above |
| band_max_2026 | excluded | as above |
| hire_date | excluded | date of joining; replaced by market_index (avoids a duplicate time trend and date-derived age signals) |
| hire_year | excluded | redundant with market_index |
| skills_count | excluded | documented as a noise variable; no job-relevant meaning |
| source | excluded | data provenance (real vs extension rows), not a job attribute |
| hire_id | excluded | identifier |
| offered_annual_fixed | target | annual fixed pay offered at hire (INR, nominal) |

**Allowed predictive features:** job_level, prior_experience_years, market_index, role_id, department_id, work_location, highest_education. **Audit-only:** `career_gap_months`, gender and age fields from `unlock_CX-15`; join/provenance fields are not predictors. **Excluded:** 2026 bands (circular), hire dates/year (time proxy/redundancy), `skills_count` (documented noise), source and IDs. Historical salary is the label, not a feature. Protected attributes are not used in prediction.

## 4. Preprocessing and evaluation strategy
`ColumnTransformer` inside each `Pipeline`: numeric = median imputation + standard scaling; categorical = constant "Unknown" imputation + `OneHotEncoder(handle_unknown="ignore")`. Fitted on **training rows only** (tested), re-used unchanged for any later row, so unseen roles/categories never crash.
Split: random 80/20, stratified by data source, **seed 42**: train 2,347, test 587. Hire IDs are unique, but no separate stable employee identifier is available to rule out repeat-person leakage; this is a row-level split, not a person-grouped or temporal holdout. The test set was not used for model selection (selection = lowest 5-fold out-of-fold MAE on training rows).

## 5. Models
* **Average-by-role baseline:** training mean salary of the role; unseen role -> overall training mean. Ignores inflation and level.
* **Linear Regression:** same pipeline, no regularisation.
* **Gradient Boosting Regression** (`GradientBoostingRegressor`, 300 trees, depth 3, learning rate 0.05, subsample 0.8, seed 42): fixed settings, **no tuning**.

## 6. Results - held-out test set (n = 587)
| Model | MAE (INR) | R² | 5-fold CV MAE (train only) | train MAE | train R² |
|---|---|---|---|---|---|
| Average-by-role baseline | 110,298 | 0.928 | 116,140 | 113,569 | 0.931 |
| Linear Regression | 82,826 | 0.961 | 84,585 | 81,168 | 0.966 |
| Gradient Boosting Regression | 60,796 | 0.978 | 65,719 | 53,964 | 0.986 |

* **Lowest MAE and best R²: Gradient Boosting Regression** (MAE 60,796, R² 0.978); it cuts MAE by 45% versus the baseline (110,298). Linear Regression changes MAE by 25% versus the baseline. Both ML models improve on this split; this alone does not establish fairness or future performance.
* **Overfitting:** Gradient Boosting's training MAE (53,964) is lower than its test MAE (60,796); the test gap is modest (ratio 1.13) and CV agrees (65,719), so mild, not severe, overfitting. Linear Regression: 81,168 train vs 82,826 test.
* **Is the data enough?** It is sufficient for a task-level benchmark on this supplied table (R² above 0.96 for both ML models; every role has at least 5 hires), but **not enough to establish production reliability**. This practice kit contains 734 real hires and 2,200 historical extension hires; results are not validated as representative of an external workforce.
* Linear Regression predicted a **negative salary for 1 test hire(s)** (linear models cannot respect "salary > 0"). Gradient Boosting and the baseline: 0 and 0. Predictions are not floored in the reported metrics; the band code never outputs a negative value.
* Charts: `reports/model_comparison.png`, `reports/salary_distribution.png`.
* Selected model feature influence: `reports/model_feature_importance.csv` (model-native split importance or absolute coefficient magnitude; one-hot categories represent individual encoded values, not causal effects).

Aggregated to the original input fields (split importance is not causal):

| input feature | aggregated model importance |
|---|---|
| job_level | 0.724 |
| department_id | 0.119 |
| role_id | 0.084 |
| market_index | 0.054 |
| prior_experience_years | 0.017 |
| highest_education | 0.001 |
| work_location | 0.001 |


## 7. Fairness testing
Method: the **official Task 12 `group_gap(scores, groups)`** is imported directly from `phase-3/task12/src/cortexhr_phase-3/task12/cortexhr_helpers.py`; that source is not modified and there is no fallback copy. It returns mean scores per group and the largest gap between group means, excluding missing labels/scores and rejecting malformed inputs. It is a warning, not proof.
It is applied to six quantities, kept separate on purpose: **actual (historical) pay**, **predicted pay**, **signed error** (predicted - actual), **absolute error**, pay relative to each role's 2026 band (an audit yardstick, never a model input), and the **recommended Low/Mid/High**.
Groups with fewer than 30 people are flagged unreliable. Missing group labels are dropped and counted, never mixed in.
**Distinctions:** a historical pay difference does not prove the model caused it; similar average predictions do not prove fairness.

### 7.1 Gender (authorized unlock_CX-15; selected model Gradient Boosting Regression; held-out test set)
| group | n (test) | reliable n | mean actual pay | mean predicted pay | mean signed error | MAE |
|---|---|---|---|---|---|---|
| Female | 246 | True | 799998 | 813156 | 13,158 | 58,808 |
| Male | 330 | True | 839974 | 829376 | -10,598 | 62,850 |
| Other | 11 | False | 836786 | 827434 | -9,352 | 43,615 |


Official `group_gap()` results:

| quantity (official group_gap) | largest gap between group means | group means |
|---|---|---|
| actual salary (historical pay) | 39,977 | Female=799,998; Male=839,974; Other=836,786 |
| predicted salary (model) | 16,220 | Female=813,156; Male=829,376; Other=827,434 |
| signed error (predicted - actual) | 23,757 | Female=13,158; Male=-10,598; Other=-9,352 |
| absolute error | 19,235 | Female=58,808; Male=62,850; Other=43,615 |
| actual pay vs role band (ratio) | 0.018 | Female=0.981; Male=0.997; Other=0.999 |
| predicted pay vs role band (ratio) | 0.017 | Female=0.997; Male=0.998; Other=0.981 |
| recommended mid | 16,220 | Female=813,156; Male=829,376; Other=827,434 |
| recommended low | 14,048 | Female=706,947; Male=720,991; Other=720,995 |
| recommended high | 17,204 | Female=914,045; Male=931,250; Other=927,536 |


The unlock file supplied 2,934 unique person references. 587 of 587 test rows had a gender label; unmatched: 0. Groups below 30 are unreliable; `Other` has n=11.
The controlled historical-pay estimate for Female vs Male is:

| group | reference | n group | n reference | pay difference % | 95% CI low | 95% CI high | CI excludes 0 |
|---|---|---|---|---|---|---|---|
| Female | Male | 1222 | 1652 | -1.50 | -2.09 | -0.91 | True |
| Other | Male | 60 | 1652 | 0.53 | -1.60 | 2.13 | False |


The allowed features' cross-validated AUC for Female vs non-Female is 0.504. This is only a binary proxy diagnostic, not proof that proxies are absent.

### 7.2 Career gap (from `salary_training.career_gap_months`; selected model Gradient Boosting Regression; test set)
| group | n (test) | mean actual pay | mean predicted pay | mean signed error | signed error % of actual | MAE | actual pay / role band | predicted pay / role band | mean rec. low | mean rec. mid | mean rec. high |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Career gap 12+ months | 40 | 610232 | 661041 | 50,809 | 7.553 | 70,196 | 0.956 | 1.019 | 571123 | 661041 | 747,991 |
| Career gap 3-9 months | 42 | 755579 | 750839 | -4,740 | -0.375 | 58,606 | 0.994 | 0.984 | 649417 | 750839 | 846,197 |
| No career gap | 505 | 845648 | 841298 | -4,350 | 0.961 | 60,233 | 0.993 | 0.996 | 731973 | 841298 | 944,377 |

Official `group_gap()` results (same model):

| quantity (official group_gap) | largest gap between group means | group means |
|---|---|---|
| actual salary (historical pay) | 235,416 | Career gap 12+ months=610,232; Career gap 3-9 months=755,579; No career gap=845,648 |
| predicted salary (model) | 180,257 | Career gap 12+ months=661,041; Career gap 3-9 months=750,839; No career gap=841,298 |
| signed error (predicted - actual) | 55,549 | Career gap 12+ months=50,809; Career gap 3-9 months=-4,740; No career gap=-4,350 |
| absolute error | 11,590 | Career gap 12+ months=70,196; Career gap 3-9 months=58,606; No career gap=60,233 |
| actual pay vs role band (ratio) | 0.038 | Career gap 12+ months=0.956; Career gap 3-9 months=0.994; No career gap=0.993 |
| predicted pay vs role band (ratio) | 0.035 | Career gap 12+ months=1.019; Career gap 3-9 months=0.984; No career gap=0.996 |
| recommended mid | 180,257 | Career gap 12+ months=661,041; Career gap 3-9 months=750,839; No career gap=841,298 |
| recommended low | 160,850 | Career gap 12+ months=571,123; Career gap 3-9 months=649,417; No career gap=731,973 |
| recommended high | 196,387 | Career gap 12+ months=747,991; Career gap 3-9 months=846,197; No career gap=944,377 |

Small groups: 40 and 42 people in the two gap groups - above the 30-person threshold but still small; treat numbers as indicative.

**Reading it.** The **raw** historical-pay gap (235,416 INR) is not a like-for-like comparison (it controls for no role, level or experience differences, and was not decomposed here), so the role-band ratio and the controlled estimate below are the relevant measures.
Relative to each role's band, hires with a 12+ month gap were historically paid **0.956x** the band versus **0.993x** without a gap, whereas the model predicts **1.019x** versus **0.996x**:
the model does **not** reproduce the penalty. Its mean signed error for the 12+ group is **+7.6%** of actual pay (versus +1.0% without a gap): the model predicts more than they were historically paid, and the **MAE is +17% higher** for them. Chart: `reports/fairness_gap.png`.

## 8. Historical pay bias: "if women (or career-gap hires) were paid less, should the model copy that?"
No. What the data shows (controlled for role, level, department, experience, location, education and hire-year price level; `reports/historical_pay_gap_controlled.csv`):

| group | vs | n group | n reference | pay difference % | 95% CI low | 95% CI high | CI excludes 0 |
|---|---|---|---|---|---|---|---|
| Career gap 12+ months | No career gap | 199 | 2549 | -4.00 | -4.99 | -2.86 | True |
| Career gap 3-9 months | No career gap | 186 | 2549 | -1.11 | -2.44 | 0.21 | False |
| Female | Male | 1222 | 1652 | -1.50 | -2.09 | -0.91 | True |
| Other | Male | 60 | 1652 | 0.53 | -1.60 | 2.13 | False |

* **Career gap, 12+ months: historically paid 4.0% less** (95% bootstrap interval -5.0% to -2.9%, excludes zero). 3-9 months: -1.1% (interval -2.4% to 0.2%, includes zero - not distinguishable from no effect). Descriptive, not causal.
* Female hires were paid 1.5% less than Male hires after controlling for job factors (95% bootstrap interval -2.1% to -0.9%; descriptive, not causal).
* **Is historical salary a feature? No** (`[]`); it is the *label*. **Gender / career gap as features? No** (`[]`, `[]`). The 2026 bands are excluded as circular.
* **Do other features act as proxies?** Allowed features predict career gap with cross-validated **AUC 0.49** (0.50 = chance); this test did not detect a strong proxy signal, but cannot prove proxies are absent. The Female-vs-non-Female proxy diagnostic AUC is 0.50; it is binary and does not establish that all gender proxies are absent.
* **Is excluding the attribute sufficient?** Not in general: the **labels** still carry the penalty, so the model is trained on, and scored against, biased history. Here the model happens not to reproduce it (above), but any accuracy measured against historical pay partly rewards copying inequity: the 12+ group looks "worse predicted" only *because* the model refuses to copy a penalty.

## 9. Mitigation experiments (original kept as M0; each variant re-trained and scored against the original historical test labels)
| variant | CV MAE (train only) | test MAE | test R² | label factors | career_gap: signed-error gap (INR) | career_gap: predicted-vs-role-band gap (ratio) | gender: signed-error gap (INR) | gender: predicted-vs-role-band gap (ratio) |
|---|---|---|---|---|---|---|---|---|
| M0 original (historical labels) | 65,719 | 60,796 | 0.978 | - | 55,549 | 0.035 | 23,757 | 0.017 |
| M1 label correction for career-gap pay difference | - | 60,746 | 0.978 | Career gap 12+ months: x1.0425; Career gap 3-9 months: x1.0132 | 58,911 | 0.036 | 24,083 | 0.014 |
| M2 drop work_location and highest_education | 64,803 | 60,250 | 0.978 | - | 57,390 | 0.037 | 22,714 | 0.013 |

* **M1 label correction** (estimate the career-gap pay difference on training rows only and add it back to those labels; features unchanged; uses the audit attribute at training time only): test MAE 60,746 vs 60,796; career-gap signed-error gap 58,911 vs 55,549; gender signed-error gap 24,083 vs 23,757. Its small MAE improvement did not improve the career-gap error gap, and it requires protected audit data during label construction: **not adopted**.
* **M2 drop `work_location` and `highest_education`** (possible proxies): CV MAE 65,719 -> 64,803 and test MAE 60,796 -> 60,250. Career-gap signed-error gap changes from 55,549 to 57,390 INR; gender signed-error gap changes from 23,757 to 22,714 INR. These metrics do not establish a universal fairness improvement. **Not adopted**: location can be a legitimate cost-of-labour factor and evidence is insufficient to remove it automatically.
* No individual recommendation was adjusted using gender or career-gap status, and group parity was not forced. **Neither M1 nor M2 is adopted**: M1's label adjustment uses protected audit information in training and did not justify its trade-off; M2 requires an HR decision about legitimate job-related factors.

## 10. Salary-band construction (`reports/salary_band_recommendations.csv`)
**Mid** = Gradient Boosting Regression prediction. **Low / High = Mid x the 10th / 90th percentile of (actual / predicted)**, taken from **training out-of-fold** residuals (2,347 rows) and computed **per job level** (a job attribute); a global fallback is used only when a level has too few training residuals to calibrate independently.
Rounded to whole rupees; guaranteed 0 <= Low <= Mid <= High. Test labels are used only to **measure** coverage. Gender and career gap never enter.

| job level | low ratio | high ratio | training residuals | test n | test coverage |
|---|---|---|---|---|---|
| 1 | 0.864 | 1.178 | 632 | 162 | 0.809 |
| 2 | 0.888 | 1.122 | 801 | 196 | 0.821 |
| 3 | 0.862 | 1.112 | 551 | 143 | 0.762 |
| 4 | 0.880 | 1.112 | 277 | 71 | 0.789 |
| 5 | 0.826 | 1.125 | 86 | 15 | 0.867 |

**Held-out coverage: 80.1%** (nominal 80%), 10.2% below Low, 9.7% above High, mean relative width 26.2%. This is an **empirical prediction band, not a proven confidence interval**; levels with held-out coverage below 78%: level 3 (76.2%, n=143); the level with the fewest test hires (5) has only 15 test hires, so its figure is noisy

## 11. Output files (`reports/`)
`salary_band_recommendations.csv` (587 held-out hires: `record_id, role_id, predicted_salary_mid, recommended_salary_low, recommended_salary_high, model_version, human_review_required, model_note`; no protected data),
`role_band_reference_2026.csv` (typical-profile band per role at the April-2026 price level), `model_comparison.csv`, `model_feature_importance.csv`,
`fairness_career_gap_groups.csv`, `fairness_career_gap_official_group_gap.csv`, `fairness_gender_groups.csv`, `fairness_gender_official_group_gap.csv`, `historical_pay_gap_controlled.csv`, `mitigation_comparison.csv`, `feature_policy.csv`, `dataset_audit.json`, `dataset_columns.csv`,
`salary_distribution.png`, `model_comparison.png`, `fairness_gap.png`, `final_validation_report.json`, `test_summary.json`; fitted pipelines in `reports/model_artifacts/` (`paybands-v1_*.joblib`). The older `models/` artifacts are left untouched.
Per-record baseline, Linear Regression and Gradient Boosting prediction/error tables (including historical salary labels) are stored only under the git-ignored `reports/private/` directory.

## 12. Validation and tests
Output checks (`final_validation_report.json`): 19 of 19 pass, including the required official gender audit, the production CSV schema and band ordering.
**Tests: 32 passed, 0 failed, 0 skipped** (`32 passed in 162.49s (0:02:42)`).

## 13. Limitations
* Gender groups include n=11 for Other, too small for a reliable standalone conclusion; age is present in the unlock file but was not a required fairness dimension and was not tested.
* **Historical labels** encode past decisions, including the measured career-gap penalty; accuracy against them is not the same as fairness.
* Practice-kit data: 734 real hires and 2,200 extension hires; external representativeness is unestablished. Rare roles have few examples (minimum 5 hires).
* Group tests have small samples (40 / 42 test hires in the gap groups); a measured gap can be noise.
* `work_location` and `highest_education` remain in the model (see M2); `market_index` assumes 5.5% yearly pay inflation; the band is a level-wise empirical band, not a confidence interval, and under-covers at some levels (section 10).
* Linear Regression can output negative salaries; only the CV-selected model (Gradient Boosting Regression) feeds the bands. The random row split is neither person-grouped nor time-based because a separate stable person identifier was unavailable.
* Recommendations were generated for the held-out test hires (out-of-sample) and as role-level references; new candidates need the same feature columns and should be assessed under a fresh temporal and employee-grouped validation.

## 14. Responsible use
A model-generated salary recommendation is **decision support, not a final compensation decision.** Final compensation must be reviewed by HR and an authorised manager; the model must not be the sole basis for a pay decision.
Do not use gender, career gap or any proxy to adjust a recommended range. Re-run the fairness audit when data or features change.

## 15. Run
```
cd task15
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Set one approved salary-training source (CSV takes precedence when both are set):
$env:CORTEXHR_SALARY_CSV = "C:\approved-data\salary_training.csv"
# Or use the master workbook, which must contain the salary_training sheet:
$env:CORTEXHR_WORKBOOK = "C:\approved-data\CortexHR_Dataset.xlsx"
# Optional for training; required to run the official protected-attribute fairness audit:
# provide the approved file at private/unlock_CX-15_protected_attributes.csv (never commit it)
python scripts/run_task15.py
python scripts/run_checks.py
python scripts/build_readme.py
```

On a clean clone, no salary data is available, so the data-backed tests and pipeline cannot run until an approved source is supplied. The protected-attribute file is confidential, remains in the Git-ignored `private/` directory, and must not be copied into reports or committed. Use a separate environment for each task; the source includes explicit skips for data-backed tests when the salary dataset is absent.
