# Model Card: Pay-band recommendation (`paybands-v1`)

## 1. Name / version
`paybands-v1`, selected model GradientBoostingRegressor (300 estimators, learning rate 0.05, depth 3, subsample 0.8, seed 42) with Linear Regression and an average-by-role baseline for comparison. Library: scikit-learn 1.8.0 (re-run).

## 2-4. Overview, use, out of scope
Suggests a mid/low/high salary range for a role from job features. Decision support only. **Not for setting an individual's pay**; not valid as evidence that past pay was fair.

## 5. Dataset
`salary_training`: 2,934 historical offers; stratified 80/20 split (2,347 train / 587 test), seed 42. Allowed features: job_level, prior_experience_years, market_index, role_id, department_id, work_location, highest_education. Audit-only (never inputs): career_gap_months, gender, age_years, age_band. Fictional data. The split is random by hire id; no stable employee id exists to rule out repeat-person leakage.

## 7. Evaluation (re-run; test set 587 rows)
| Model | MAE | R2 | CV MAE (train only) |
|---|---|---|---|
| Average-by-role baseline | 110,298 | 0.928 | 116,140 |
| Linear Regression | 82,826 | 0.961 | 84,585 |
| **Gradient Boosting** | **60,801** | **0.978** | 65,652 |
Your supplied report shows 60,796 for gradient boosting; the 5-unit difference is consistent with library-version effects. Empirical 80% band coverage on the test set 0.802 (below-low 0.101, above-high 0.097, mean relative width 0.259).

## 8. Fairness evaluation (official audit now run with `unlock_CX-15`)
Test groups: Female 246, Male 330, Other 11 (**too small to interpret**). Selected model, mean predicted salary: Female 813,171, Male 829,381; actual historical pay Female 799,998 vs Male 839,974. Signed error (pred - actual): Female +13,173, Male -10,593. Career gap: predicted 12+ months 661,112 vs none 841,303; signed error +50,880 for 12+ months (the model does not reproduce the full historical gap 610,232 vs 845,648). Controlled historical pay difference: career gap 12+ months **-4.0%** (CI -5.0, -2.9); women **-1.5%** (CI -2.1, -0.9). Mitigations tried: label correction and dropping location/education changed the career-gap error gap from 55,619 to 58,911 and 55,463, i.e. **no meaningful improvement**. Predicted-pay-versus-band ratio gap (gender) 0.017.

## 9. Limitations
Trained on historical offers that contain a measurable pay gap; the model never uses gender but cannot remove gaps embedded in role/level history; small "Other" group; random split; point estimate quality varies by role. The API returns **null low/high bands** unless a calibration file is configured (midpoint only).

## 10. Ethical considerations
A model-generated figure can anchor negotiations and entrench inequity. Every row is flagged for human review; compensation decisions need an accountable HR/finance owner.

## 11. Deployment
`POST /v1/payband/analyze`, verified locally (midpoint returned; `band_status: unavailable: calibration file not configured`). Not tested remotely.

## 12. Monitoring
Task 18 re-runs the pay-gap audit against the Task 15 baseline: gender gap 0.0170 (baseline 0.0170), career gap 0.0348 (0.0350), status OK.

## 13. Release recommendation
**RELEASE WITH WARNINGS** for demonstration; **not recommended for real compensation use** (see Manager Release Decision).
