# Model Card: CortexHR attrition model V2 (`attrition-v2`)

## 1. Model name / version
`attrition-v2`, Logistic Regression (balanced) wrapped in sigmoid calibration (`CalibratedClassifierCV`, 5 inner folds). Metadata: scikit-learn 1.8.0, numpy 2.4.4, pandas 3.0.2, Python 3.12.3. The artifact `trained_at_utc` in the supplied metadata is 2026-11-02; the artifact used in this audit was **regenerated locally on 2026-11-20** (the supplied `model.joblib` was corrupted in upload) and reproduces every metric exactly.

## 2. Overview
Estimates the probability that an employee leaves within 12 months of an observation date (`left_within_12m`). Intended users: HR analysts producing a ranked, explained watch-list for retention conversations.

## 3. Intended use
Rank-based prioritisation for human-led retention support; aggregate workforce planning. Scores are probabilities calibrated on out-of-fold data.

## 4. Out-of-scope use
Not for terminating, demoting, denying promotion or changing pay; not a prediction of individual intent; not for use without human review; not valid for populations unlike the training data.

## 5. Dataset
Workbook sheet `attrition_dataset`: 2,389 raw rows -> 2,379 trained (6 duplicates dropped, 4 negative-tenure rows excluded), 349 leavers (14.7%). 33 features (listed in `metadata.json`). Excluded as leakage: exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount, voluntary_left_within_12m, ids, `dept_attrition_rate_prior_12m`. **The data is fictional/synthetic.** Max single-feature AUC 0.601; label-shuffle AUC 0.471 (leakage audit, re-run).

## 6. Model and methodology
Task 2 pipeline (cleaning, median impute/scale, one-hot) fitted inside the estimator. `class_weight="balanced"`, `max_iter=2000`, seed 42. Validation: the 5 stratified folds of Task 3 (fingerprint `646e474fae66cb0c`). A tuned LightGBM challenger was evaluated and **not shipped**: mean ROC-AUC gain 0.0105, paired 95% interval for the pooled gain [-0.016, 0.031] includes zero.

## 7. Evaluation (out-of-fold, re-run in this audit; all 76 numeric metrics identical to the shipped file)
| Metric | Value |
|---|---|
| ROC-AUC (per-fold mean +/- sd) | 0.7114 +/- 0.0182 |
| ROC-AUC pooled (95% CI) | 0.708 (0.677 - 0.736) |
| PR-AUC | 0.3097 (prevalence 0.147) |
| Precision / recall @ top 10% | 0.332 / 0.226 (lift 2.26) |
| Threshold 0.5: precision / recall / F1 / accuracy | 0.327 / **0.023** / 0.040 / 0.854 |
| Brier (vs 0.1252 constant) | 0.1163 (skill 0.071) |
| ECE (10 quantile bins) | 0.020; calibration slope 1.12, intercept 0.18 |
| Baseline LR balanced (uncalibrated) ROC-AUC | 0.7121 |
Probability range on the 430-row 2026 scoring file: 0.006 - 0.491 (mean 0.128). **A fixed threshold of 0.5 flags almost nobody; use rank-based bands.**

## 8. Fairness evaluation
**No fairness evaluation of this model was performed.** Gender, age and similar attributes are not model inputs, but proxies (e.g. tenure, experience, work mode) were not tested for group disparity. `total_experience_years` is an age-correlated input; Task 14 suppresses it in explanations. Status: NOT TESTED.

## 9. Limitations
Modest discrimination (AUC ~0.71); synthetic data; one time window; no external validation; the 12-month label depends on the data generator; API accepts records missing the three monthly-derived features (they are then median-imputed, see Service Verification).

## 10. Ethical considerations
Attrition scores about individuals can be misused for adverse treatment or surveillance. Share only with trained HR staff, never with line managers as a ranking of "flight risks", and never as sole basis for any employment action.

## 11. Deployment information
`POST /v1/attrition/predict` in `cortexhr_api` (Task 17). Verified **locally only** (FastAPI TestClient): 200 on 25 real scoring rows, loads `attrition-v2` metadata. Not deployed or tested remotely. Requires scikit-learn 1.8.x.

## 12. Monitoring
Task 18 monitor computes PSI on 8 inputs. 2026-11-20 run: 6 OK, 2 WATCH (`tenure_months` 0.120, `attendance_rate_3m` 0.039 vs matched reference; **1.467 vs the full training set**, flagged as training-serving skew). No outcome-based performance monitoring exists (no new labels).

## 13. Release recommendation
**RELEASE WITH WARNINGS (demonstration/decision-support only).** Reproducible and well calibrated, but discrimination is modest, fairness is untested, and the data is synthetic.
