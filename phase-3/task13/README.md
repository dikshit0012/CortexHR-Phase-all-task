# Task 13 - Attrition Model V2: beat the baseline honestly

*Phase 3 (Intelligence), Day 13. Part of the CortexHR AI/ML employee-attrition project. Every number below is produced by `run_task13.py` and stored in `reports/`; none is typed in by hand.*

## Result in one paragraph

A tuned LightGBM model reaches a cross-validated ROC-AUC of **0.723** (nested CV, honest) against **0.712** for the Task 3 balanced Logistic Regression, on the identical five folds. That is a gain of **+0.010** ROC-AUC per fold (better in 4 of 5 folds), but the paired bootstrap 95% interval for the pooled difference is **[-0.016, +0.031]**, which includes zero. On PR-AUC the gain is +0.041 with an interval of [+0.006, +0.074]. By the selection rule written down before the final numbers were read, the gain is **not reliable enough to justify the harder-to-explain model**, so the shipped `attrition-v2` is the **calibrated balanced Logistic Regression**. Its real improvement over the Task 3 model is not accuracy of ranking, it is **honest probabilities**: calibration error (ECE) falls from 0.259 to 0.020. The gradient-boosting model remains a documented challenger.

> **Completion decision:** Task 13 is complete under its predeclared promotion rule. LightGBM was trained, tuned, calibrated, compared on the shared folds, and retained as a challenger; it was not promoted because the ROC-AUC interval includes zero. The versioned `attrition-v2` artifact remains calibrated Logistic Regression. Do not interpret the challenger’s point estimate as proof of a real gain or change the deployed model without new validation evidence.

## 1. Project objective

Build a stronger employee-attrition model with gradient boosting (LightGBM), compare it with the existing baselines on **identical cross-validation folds**, tune it without overfitting, check whether its probabilities are trustworthy, and save the final model as `attrition-v2` with the complete preprocessing pipeline. If V2 does not clearly win, say so.

## 2. Dataset

| | |
|---|---|
| Workbook / sheet | `02_CortexHR_AI_ML_Dataset.xlsx` (the delivered `CortexHR_Dataset.xlsx`), sheet `attrition_dataset` |
| Raw shape | 2,389 rows x 42 columns |
| Rows used | **2,379** (6 exact duplicates dropped, 4 negative-tenure rows held out, Task 1 decisions D01/D03, open item O1) |
| Input columns | **33** (6 categorical, the rest numeric), 3 of them monthly attendance/payroll features from Task 2 |
| Sources | 2,000 simulated historical rows + 379 master-snapshot employees (the workbook is fictional, per its README sheet) |
| Missing values | structural, mostly: monthly features blank for 84% of rows (all simulated history rows), months-since-promotion 61% (never promoted), engagement 22% |
| Duplicates left | 0 duplicate employee ids among the rows used |

## 3. Target variable

`left_within_12m`, **verified in the workbook** (not assumed): 349 leavers and 2,030 stayers among the rows used = **14.7% leave** (about 5.8 stayers per leaver). Because of this imbalance the headline metrics are ROC-AUC and PR-AUC, not accuracy (a model that always says "stays" is 85.3% accurate and catches nobody).

## 4. Leakage audit

The Task 2 pipeline refuses the columns below (raises `ForbiddenColumnError`), and Task 13 re-checks it (`data_validation.assert_no_leakage_columns`):

| column | category | reason |
|---|---|---|
| left_within_12m | target | it is the target (the answer we are predicting) |
| exit_date | leakage (exists only after exit) | it leaks the outcome: it only exists after the person has left (Task 1, D18/D19) |
| resignation_date | leakage (exists only after exit) | it leaks the outcome: it only exists after the person has left (Task 1, D18/D19) |
| exit_reason | leakage (exists only after exit) | it leaks the outcome: it only exists after the person has left (Task 1, D18/D19) |
| exit_interview_done | leakage (exists only after exit) | it leaks the outcome: it only exists after the person has left (Task 1, D18/D19) |
| fnf_settlement_amount | leakage (exists only after exit) | it leaks the outcome: it only exists after the person has left (Task 1, D18/D19) |
| voluntary_left_within_12m | leakage (exists only after exit) | it leaks the outcome: it only exists after the person has left (Task 1, D18/D19) |
| row_id | identifier / non-feature | it is a row counter, not an employee trait |
| employee_id | identifier / non-feature | it is an identifier (use it only to group or join) |
| source | identifier / non-feature | it records which simulation produced the row, not anything about the employee |
| observation_date | identifier / non-feature | it is the prediction date; use it for time windows and splits, not as a feature |
| dept_attrition_rate_prior_12m | unconfirmed (open item O2) | not confirmed to use only exits before the observation date; off by default |

Additional checks run in this task (`reports/leakage_single_feature_auc.csv`):

* **No single input is a giveaway:** the best one-column cross-validated ROC-AUC is 0.601 (`last_engagement_score`); the alarm threshold is 0.9.
* **Label-shuffle control:** the V2 model trained on shuffled labels scores ROC-AUC **0.471** (chance is 0.5).
* **Feature importance looks plausible, not leak-shaped:** top gain shares are last_engagement_score (22%), months_since_last_raise (20%), commute_distance_km (8%) (`reports/attrition_v2_feature_importance.csv`).
* **New features:** none were introduced. Nothing was added to Task 2's list, so nothing new needed an availability-at-prediction-time check.
* **Still unconfirmed (open item O2):** `dept_attrition_rate_prior_12m` stays off. Sensitivity run (reported only, never used to decide): LR balanced without it: 0.712; LR balanced with it: 0.710; LightGBM E09 without it: 0.727; LightGBM E09 with it: 0.726. It adds nothing either way.
* **A known shortcut, not a leak:** the 2,000 simulated history rows have no monthly data, so after the median fill a model can partly recognise the data source (see Limitations and `reports/attrition_v2_per_source_auc.csv`).

## 5. Task 2 pipeline reuse

`cortexhr.baselines.model_pipeline(classifier)` wraps the unchanged Task 2 `build_pipeline()` and the classifier into **one** scikit-learn `Pipeline`. Imputation, scaling, the 10x-salary correction, one-hot encoding and the monthly features are therefore **re-fitted on the training part of every fold** and only applied to the validation part. Task 13 imports the shared package from `phase-1/task04/src` (override with `CORTEXHR_ML_SRC`); no Task 1-12 file was modified by this task.

## 6. Task 3 fold reuse

Task 3 did not save fold indices. The folds are rebuilt with Task 3's own function `cortexhr.baselines.make_folds(y)`: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over the 2,379 prepared rows. `data_validation.verify_folds` confirms: 5 folds, every row tested exactly once, no train/test overlap, test sizes [476, 476, 476, 476, 475], leaver rates [0.1471, 0.1471, 0.1471, 0.1471, 0.1453], and equality with an independent `StratifiedKFold` rebuild. Fingerprint `646e474fae66cb0c`; the indices are saved in `reports/cv_folds.json`. **All models in this report are scored on these folds, and the baselines reproduce Task 3** (LR balanced 0.712 +/- 0.018, as in the Task 3 report). The folds were never changed.

## 7. Gradient boosting (LightGBM)

Gradient boosting builds many small decision trees one after another; each new tree tries to fix the mistakes of the ones before it. It can capture non-linear effects and interactions that a logistic regression cannot, but it needs more care on small data: this project has only 349 leavers, so deep trees memorise. The brief's starting values (200 trees, depth 5, 15 leaves) reach a training ROC-AUC of 0.997 but only 0.709 in cross-validation, a textbook overfit; that is why the tuning below favours shallow, slow, regularised trees. `class_weight='balanced'` is used so the 0.5-cut-off precision/recall are comparable with Task 3's balanced baselines; calibration (section 12) restores honest probabilities. LightGBM was chosen because it works directly on the Task 2 output with no changes.

## 8. Hyperparameter tuning methodology

* **13 experiments** only (`E00`-`E12`), varying learning rate, trees, depth, leaves, minimum child samples, subsampling and L1/L2 regularisation. No random search, no second round.
* Each is scored on the five shared folds (`reports/attrition_v2_tuning_log.csv`). **Choice rule: one standard error.** Pick the simplest configuration (fewest leaves x trees) whose mean ROC-AUC is within one standard error of the best. This avoids rewarding a lucky fold: the highest mean was `E12`, the selected one is `E09`.
* **Honest estimate: nested cross-validation.** Because the same folds chose the winner, its score is optimistic. So the tuned model is also evaluated with nested CV: inside each Task 3 training part a separate 5-fold split makes the choice, and the Task 3 test part is only scored. Choices per outer fold: ['E09', 'E09', 'E02', 'E10', 'E09'].
* There is no separate hold-out test set in this project (Tasks 3 and 4 also use cross-validation only); no test data influenced tuning.

## 9. Tuning log

| ID | Configuration | ROC-AUC | sd | PR-AUC | F1 | Train ROC-AUC | Leaves x trees | Selected |
|---|---|---|---|---|---|---|---|---|
| E00 | brief defaults (reference) | 0.7086 | 0.0211 | 0.3118 | 0.3279 | 0.9970 | 3000 | False |
| E01 | shallower: depth 3, 7 leaves, min child 30, lr 0.03, 300 trees | 0.7222 | 0.0181 | 0.3396 | 0.3794 | 0.9376 | 2100 | False |
| E02 | depth 2, 4 leaves, min child 40, lr 0.05, 200 trees | 0.7218 | 0.0200 | 0.3492 | 0.3765 | 0.8721 | 800 | False |
| E03 | E01 + L2 reg 5 + min child 50 | 0.7210 | 0.0164 | 0.3391 | 0.3879 | 0.9117 | 2100 | False |
| E04 | depth 4, 8 leaves, slow lr 0.02, 400 trees, L2 10 | 0.7198 | 0.0169 | 0.3364 | 0.3945 | 0.9256 | 3200 | False |
| E05 | E01 with 100 trees, lr 0.05 (fewer, faster) | 0.7216 | 0.0214 | 0.3429 | 0.3800 | 0.8937 | 700 | False |
| E06 | very shallow: depth 2, 3 leaves, min child 60, lr 0.03, 400 trees, L2 5 | 0.7280 | 0.0177 | 0.3510 | 0.3782 | 0.8404 | 1200 | False |
| E07 | E01 with colsample 0.5, subsample 0.7 | 0.7227 | 0.0186 | 0.3455 | 0.3847 | 0.9314 | 2100 | False |
| E08 | brief shape (depth 5, 15 leaves) but slow lr 0.02, 300 trees, L1 1, L2 10 | 0.7169 | 0.0172 | 0.3308 | 0.3902 | 0.9582 | 4500 | False |
| E09 | decision stumps (depth 1, 2 leaves): additive model, lr 0.05, 300 trees | 0.7265 | 0.0195 | 0.3597 | 0.3826 | 0.8081 | 600 | True |
| E10 | depth 2, 4 leaves, min child 30, lr 0.03, 150 trees | 0.7260 | 0.0196 | 0.3456 | 0.3827 | 0.8296 | 600 | False |
| E11 | depth 3, 7 leaves, min child 30, lr 0.01, 500 trees, L2 5, colsample 0.6 | 0.7267 | 0.0156 | 0.3490 | 0.3913 | 0.8782 | 3500 | False |
| E12 | stumps, slower: depth 1, lr 0.02, 500 trees, L2 5 | 0.7287 | 0.0206 | 0.3620 | 0.3799 | 0.7847 | 1000 | False |

Same-fold mean of `E09`: 0.7265 (optimistic). Nested estimate of the whole "tune then pick" procedure: **0.7226**.

## 10. Model comparison (same five folds for every row)

Per-fold mean and standard deviation. Precision, recall, F1, accuracy use the 0.5 cut-off (as Task 3); "Precision @ top 10%" is the share of leavers in the 10% highest-scored people (Task 4's review-list view).

| Model | Kind | ROC-AUC mean | ROC-AUC std | PR-AUC mean | F1 mean | Precision | Recall | Accuracy | Precision @ top 10% |
|---|---|---|---|---|---|---|---|---|---|
| Dummy (always stays) | reproduced_baseline | 0.500 | 0.000 | 0.147 | 0.000 | 0.000 | 0.000 | 0.853 | 0.139 |
| Logistic Regression (unweighted) | reproduced_baseline | 0.708 | 0.020 | 0.309 | 0.154 | 0.393 | 0.097 | 0.845 | 0.328 |
| Logistic Regression (balanced) [Task 3 baseline] | reproduced_baseline | 0.712 | 0.018 | 0.309 | 0.373 | 0.265 | 0.633 | 0.686 | 0.332 |
| Decision Tree (depth 4, balanced) | reproduced_baseline | 0.654 | 0.038 | 0.255 | 0.325 | 0.236 | 0.525 | 0.681 | 0.319 |
| Random Forest (300 trees, balanced) | reproduced_baseline | 0.706 | 0.022 | 0.304 | 0.017 | 0.500 | 0.009 | 0.854 | 0.315 |
| LightGBM V2 (brief defaults, E00) | new_model | 0.709 | 0.021 | 0.312 | 0.328 | 0.305 | 0.355 | 0.786 | 0.349 |
| LightGBM tuned (config picked on the same folds; optimistic) | new_model | 0.727 | 0.019 | 0.360 | 0.383 | 0.276 | 0.630 | 0.702 | 0.399 |
| Tuned V2 (nested CV; honest) | new_model | 0.723 | 0.021 | 0.352 | 0.376 | 0.274 | 0.605 | 0.706 | 0.395 |

Paired comparison, nested V2 against the Task 3 baseline (same rows, same folds):

* ROC-AUC difference per fold: +0.013, +0.016, -0.011, +0.023, +0.012; mean +0.0105 (sd 0.0129), V2 better in 4/5 folds.
* Pooled ROC-AUC difference +0.0080, paired-bootstrap 95% interval [-0.0162, +0.0310]; paired t-test over 5 folds p = 0.14 (only 4 degrees of freedom; a rough guide).
* Pooled PR-AUC difference +0.0411, interval [+0.0061, +0.0745].
* The brief-defaults LightGBM is **not** better than the baseline (-0.0035 ROC-AUC per fold, wins 3/5): the improvement comes from tuning, not from the algorithm alone.
* Random Forest does not quite reach the benchmark's own number either (0.706); see section 11.

## 11. Official benchmark comparison

Rows are labelled `official_benchmark` (copied from the workbook's `baseline_benchmarks` sheet), `reproduced_baseline` (our run on the shared folds) and `new_model`.

| Row type | Model | ROC-AUC | sd | PR-AUC | Precision | Recall | F1 | Accuracy | Δ ROC-AUC vs official | Δ PR-AUC vs official |
|---|---|---|---|---|---|---|---|---|---|---|
| official_benchmark | Dummy (always stays) | 0.500 | 0.000 | 0.146 | 0.000 | 0.000 | 0.000 | 0.854 |  |  |
| official_benchmark | Logistic Regression (class_weight=balanced) | 0.713 | 0.010 | 0.332 | 0.260 | 0.630 | 0.368 | 0.683 |  |  |
| official_benchmark | Decision Tree (depth 4, balanced) | 0.650 | 0.019 | 0.256 | 0.234 | 0.524 | 0.322 | 0.676 |  |  |
| official_benchmark | Random Forest (300 trees, balanced) | 0.715 | 0.018 | 0.336 | 0.523 | 0.089 | 0.151 | 0.854 |  |  |
| official_benchmark | LEAKY Logistic Regression (adds exit information) | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |  |
| reproduced_baseline | Dummy (always stays) | 0.500 | 0.000 | 0.147 | 0.000 | 0.000 | 0.000 | 0.853 | 0.000 | 0.001 |
| reproduced_baseline | Logistic Regression (balanced) [Task 3 baseline] | 0.712 | 0.018 | 0.309 | 0.265 | 0.633 | 0.373 | 0.686 | -0.001 | -0.023 |
| reproduced_baseline | Decision Tree (depth 4, balanced) | 0.654 | 0.038 | 0.255 | 0.236 | 0.525 | 0.325 | 0.681 | 0.004 | -0.001 |
| reproduced_baseline | Random Forest (300 trees, balanced) | 0.706 | 0.022 | 0.304 | 0.500 | 0.009 | 0.017 | 0.854 | -0.009 | -0.032 |
| new_model | LightGBM V2 (brief defaults, E00) | 0.709 | 0.021 | 0.312 | 0.305 | 0.355 | 0.328 | 0.786 |  |  |
| new_model | LightGBM tuned (config picked on the same folds; optimistic) | 0.727 | 0.019 | 0.360 | 0.276 | 0.630 | 0.383 | 0.702 |  |  |
| new_model | Tuned V2 (nested CV; honest) | 0.723 | 0.021 | 0.352 | 0.274 | 0.605 | 0.376 | 0.706 |  |  |

* **Reproduced:** Dummy, Logistic Regression and the depth-4 tree match their official rows within about one fold-sd on ROC-AUC. The PR-AUC of the logistic regression is lower than official (0.309 vs 0.332), a gap Task 3 already explained (fewer rows after cleaning, and the gap is below two fold-sds).
* **Random Forest: not exactly reproduced.** The sheet gives only "300 trees, balanced". Ours reaches ROC-AUC 0.706 (official 0.715, a difference inside the fold-sd), but at the 0.5 cut-off our forest flags almost nobody (recall 0.009) while the official row shows recall 0.089. The other settings that differ were not documented, so this was not chased further.
* **The leaky row (ROC-AUC 1.000) is intentionally not reproduced.** It adds exit information that does not exist at prediction time.
* The new models have no official counterpart; compare them with the official logistic regression (0.713) and forest (0.715).

## 12. Calibration analysis

A model is calibrated if, among people it scores at 0.30, about 30% really leave. Method: calibration is measured on out-of-fold predictions; Platt scaling (sigmoid) and isotonic regression are fitted with `CalibratedClassifierCV` **inside each training part only** (an inner 5-fold split), so the validation rows never shape the calibration map. For the tuned V2 the whole thing is nested (configuration choice and calibration both inside the outer training part). Files: `reports/calibration_comparison.csv`, `reports/calibration_curve.png`.

| Model | calibration | Brier | Brier skill | ECE | Cal. slope | Cal. intercept | Mean predicted | Max predicted | Pooled ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| LR balanced | none | 0.2061 | -0.6465 | 0.2588 | 0.5965 | -1.6604 | 0.4055 | 0.9773 | 0.7103 |
| LR balanced | sigmoid | 0.1163 | 0.0710 | 0.0200 | 1.1211 | 0.1822 | 0.1480 | 0.6264 | 0.7080 |
| LR balanced | isotonic | 0.1169 | 0.0665 | 0.0293 | 0.8503 | -0.2497 | 0.1486 | 0.8564 | 0.7052 |
| Tuned V2 (nested) | none | 0.1983 | -0.5841 | 0.2800 | 1.1138 | -1.6331 | 0.4267 | 0.9218 | 0.7183 |
| Tuned V2 (nested) | sigmoid | 0.1138 | 0.0908 | 0.0137 | 1.0625 | 0.0957 | 0.1469 | 0.6565 | 0.7201 |
| Tuned V2 (nested) | isotonic | 0.1149 | 0.0818 | 0.0138 | 0.7783 | -0.3514 | 0.1469 | 1.0000 | 0.7125 |
| LightGBM E09, unweighted | none | 0.1121 | 0.1046 | 0.0237 | 1.1150 | 0.1802 | 0.1466 | 0.8180 | 0.7280 |

* **As trained, neither balanced model gives usable probabilities:** the balanced class weight inflates scores (mean predicted 0.41 for LR and 0.43 for V2 against a true rate of 0.147; ECE 0.26 and 0.28). Their Brier scores are worse than always predicting the base rate (Brier skill below zero).
* **Calibration fixes it:** after sigmoid scaling ECE is 0.020 (LR) and 0.014 (V2); mean predicted equals the base rate. Isotonic did not beat sigmoid by the margin required (0.002 Brier) and has more variance with ~350 leavers, so sigmoid is used.
* An **unweighted** LightGBM of the same configuration is already close to calibrated without any correction (ECE 0.024), at a slightly higher ROC-AUC (0.728, same-fold selection, so optimistic).
* **Why a high ROC-AUC does not mean good probabilities:** ROC-AUC only asks whether leavers are ranked above stayers, so any monotone rescaling of the scores (such as multiplying them by 3, or class weighting) leaves it unchanged. Calibration asks whether the *numbers* are right. The raw V2 and raw LR both rank people at ~0.71-0.72 while saying "41-43% chance of leaving on average" for a group where 14.7% leave. If HR budgets from the probabilities (expected leavers = sum of probabilities), an uncalibrated model overstates the cost about three times.

## 13. Final model selection

The rule (`evaluation.selection_decision`, fixed before the final numbers were read): ship gradient boosting only if **all** hold against the simplest strong baseline: mean per-fold ROC-AUC gain >= 0.01, paired-bootstrap 95% interval excludes zero, wins >= 4 of 5 folds, calibration not clearly worse.

| Check | Result |
|---|---|
| mean_fold_gain_at_least_0.01 | pass |
| bootstrap_ci_excludes_zero_in_favour | **fail** |
| wins_at_least_4_of_5_folds | pass |
| calibration_not_worse | pass |

Mean gain +0.0105 and 4/5 folds pass; the interval does not exclude zero, so the decision is **keep the simpler model**. *"If V2 is only slightly better but much harder to explain, would you ship it?"* No: a logistic regression's coefficients can be shown to an HR manager, a 300-tree boosted ensemble needs SHAP-style tooling, and the evidence for better ranking is not yet reliable on 349 leavers.

The honest counter-point: PR-AUC and the top-10% list do favour V2 (precision in the top 10%: 0.395 vs 0.332, PR-AUC interval above zero). That is encouraging, but the rule was fixed on ROC-AUC beforehand and it is not changed after seeing the results. It is a reason to **re-test V2 when more data is available**, not a claim of success.

**Shipped as `attrition-v2`:** balanced Logistic Regression + sigmoid calibration, trained on all 2,379 rows. Cross-validated out-of-fold evidence for this exact model: ROC-AUC (per fold) 0.711 +/- 0.018, PR-AUC 0.310, Brier 0.1163, ECE 0.020, calibration slope 1.12. In plain terms: it ranks like the Task 3 model but its percentages can now be believed.

## 14. Model versioning

```
model_artifacts/attrition-v2/
    model.joblib     fitted CalibratedClassifierCV( Task 2 pipeline + LogisticRegression ), one file, no retraining at load time
    metadata.json    name, version, training date, target, 33 input columns, excluded columns, preprocessing, hyperparameters,
                     CV strategy + fold fingerprint, calibration method, seed, dataset, library versions, decision
    metrics.json     out-of-fold evidence, paired comparison
```

`model_versioning.save_model` refuses to overwrite an existing version. `load_model()` only reads files; `predict_attrition(df)` validates columns (missing columns and forbidden columns are errors, missing *values* are imputed by the saved pipeline), checks 0 <= p <= 1, returns the model version, and logs only record counts. Scoring needs the same monthly attendance/payroll features as training: `build_scoring_table()` builds them for the unlabeled `attrition_scoring_2026` sheet (430 employees scored in the last run, probabilities 0.006-0.491, mean 0.128).

## Architecture

```
workbook (attrition_dataset + monthly sheets)
   |  Task 1/2: row rules, monthly features, safe-column list (cortexhr, unchanged)
   v
X (2,379 x 33), y = left_within_12m ----> Task 3 folds (StratifiedKFold 5, seed 42, rebuilt with make_folds)
   |                                         |
   |      every model = Pipeline[ Task 2 features -> classifier ], re-fitted per fold
   v                                         v
reference models  +  LightGBM E00..E12  --> out-of-fold scores --> comparison, paired tests, benchmark table
                          |                                   \
                nested CV (inner split chooses config)         calibration (inner CV, per outer fold)
                          v                                              v
                   selection rule  ---------------------------->  attrition-v2 (model.joblib + metadata + metrics)
```

## 15. Installation

1. Python 3.10+ and the authorized workbook at `phase-1/task04/data/raw/02_CortexHR_AI_ML_Dataset.xlsx` (not committed to git; or set `CORTEXHR_WORKBOOK`).
2. Windows PowerShell:
   ```powershell
   cd "task 13"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   Linux / macOS: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## 16. Execution

```powershell
python run_task13.py                      # about 3 minutes on one CPU core; writes reports/ and model_artifacts/attrition-v2
python run_task13.py --version attrition-v2-rerun   # an existing version is never overwritten
python -m pytest tests -q                 # 21 tests (needs the run above for the artifact)
```

```python
import sys; sys.path.insert(0, "src")
from attrition_v2 import model_versioning as mv
loaded = mv.load_model()                           # attrition-v2
scores = mv.predict_attrition(mv.build_scoring_table(), loaded)   # probability of leaving within 12 months
```
(The artifact is a pickle: only load model files you trust. It needs scikit-learn 1.8.0 or a compatible version and the `cortexhr` package on the path; importing `attrition_v2` does that.)

## How I verified it works

* `python run_task13.py` executed end to end (187 s); a second run gave identical numbers (deterministic seeds, single thread).
* 21 new tests in `tests/test_attrition_v2.py` pass (dataset, target, leakage exclusion, fold identity, training determinism, tuning rule and log, nested CV, metric maths, benchmark source, selection rule, calibration, serialisation, loading, prediction on new records, probability range, missing values, version id, no values in logs). The 110 existing Task 1-4 tests still pass.
* The baselines reproduce Task 3 (0.712 ROC-AUC) before anything new is claimed.

## 17. Limitations

* **Small, simulated data.** 349 leavers; the fold-to-fold sd of ROC-AUC (~0.02) is larger than the gain being tested. Intervals are wide; "not significant" here also means "not enough data to tell".
* **Random folds, not time-based.** Rows come from two sources with different dates. A future-period test would be stricter. Task 3's folds were kept on purpose.
* **Source shortcut.** History rows lack monthly data and are filled with a constant, which identifies them. Within-source ROC-AUC (`attrition_v2_per_source_auc.csv`): nested V2 0.722 / 0.701 (history / master) vs LR 0.714 / 0.688. The ordering holds in both groups, so the shortcut does not explain the gain, but it is not removed.
* **Open items O1/O2** (negative tenure dates, `dept_attrition_rate_prior_12m`) are still unanswered; the 4 rows stay out and the feature stays off.
* **Calibration is estimated on the same 2,379 rows** by cross-validation; the saved model's calibration should be re-checked on new data.
* **Random Forest benchmark row not exactly reproduced** (section 11); the leaky benchmark row is not reproducible by design.
* **No API code was in the uploaded repository,** so no API was changed or tested; `predict_attrition` / `load_model` are the integration points, and nothing retrains at start-up.
* The pickle format ties the artifact to compatible library versions.

## 18. Future improvements

* Re-run the V2 vs baseline test when more leavers are available, or with repeated cross-validation (extra seeds) to shrink the uncertainty; promote V2 only if the rule passes.
* Time-based validation; monthly features for the history rows so the source shortcut disappears.
* If V2 is promoted: SHAP explanations for HR, and monotonic constraints on features such as tenure and engagement.
* Answer O2 and then include `dept_attrition_rate_prior_12m` in both models and re-test.
* Decide the operating point with Task 4's cost matrix applied to the now-calibrated probabilities.

## 19. Set up from this repository

The Task 2-4 `cortexhr` package is included at `phase-1/task04/src`. The authorized master workbook is an external prerequisite and is intentionally not committed. Set `CORTEXHR_ML_SRC` only if you use a different package location, and set `CORTEXHR_WORKBOOK` to the authorized workbook. It must contain the `attrition_dataset`, attendance, and payroll sheets used by Task 2.

Task 13 creates the `attrition-v2` artifact under the Git-ignored `model_artifacts` directory. Task 14 loads that artifact and requires scikit-learn 1.8.x compatibility, so build it with the version pinned in this task's requirements.

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv-task13
.\.venv-task13\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r phase-3\task13\requirements.txt

$env:CORTEXHR_ML_SRC = 'C:\approved-source\cortexhr-ml\src'
$env:CORTEXHR_WORKBOOK = 'C:\approved-data\02_CortexHR_AI_ML_Dataset.xlsx'

python -m pytest phase-3\task13\tests
python phase-3\task13\run_task13.py
```

Without the external source package, the module fails explicitly during import. Without the workbook, the data-backed tests and training run cannot proceed; no substitute dataset is generated. The local model and generated reports must not be committed.
