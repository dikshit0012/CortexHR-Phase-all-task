# Final Challenge: input-completeness guard for attrition scoring

The experiment source and tests are in `phase-4/task20/`; results are in `docs/final_challenge/results/experiment_results.json`. The guard remains a standalone demonstration and is not wired into the production API.

## Problem
The audit found that `attrition-v2` median-imputes any missing feature, and `POST /v1/attrition/predict` accepts records that omit the three monthly-derived features (`attendance_rate_change_3m`, `avg_unpaid_absence_days_3m`, `avg_variable_pay_3m`). The delivered `attrition_scoring_2026.csv` omits all three for **every one of its 430 rows**, so any caller who scores that file as-is gets scores built on partly missing evidence, with no warning.

## Motivation and hypothesis
Chosen because it is a real weakness discovered by the audit, is independent of production code, and is measurable. Hypothesis: silent imputation lowers ranking quality, and a guard that withholds scores for incomplete records would improve the quality of what is shown.

## Baseline (preserved; no production file was changed)
Same Task 3 folds (seed 42), Logistic Regression (balanced) on all 33 features, out-of-fold:
ROC-AUC 0.7103 (per-fold mean 0.7121), PR-AUC 0.298, precision@top10% 0.332, recall@top10% 0.226.

## Decision and implementation
A pure function module `completeness_guard.py`: `check_record` / `check_batch` mark a record `review` when any monthly-derived feature is missing or more than 10% of features are missing, and return the list of missing fields. It imputes nothing and changes no score. Six unit tests pass.

## Results
| Scenario (out-of-fold, 2,379 rows) | ROC-AUC | PR-AUC | Precision@10% | Recall@10% | Notes |
|---|---|---|---|---|---|
| A. all records complete | 0.7103 | 0.298 | 0.332 | 0.226 | baseline |
| B. all 3 monthly features missing, imputed | 0.7107 | 0.299 | 0.332 | 0.226 | no accuracy loss |
| C. 30% of rows missing, score everyone | 0.7104 | 0.298 | 0.332 | 0.226 | no accuracy loss |
| C. strict guard: score only the 70% complete | 0.7184 | 0.304 | 0.343 | 0.241 | coverage 70%; **112 of 349 leavers (32%) get no score** |
| C. the 714 incomplete rows only (baseline scores) | 0.6925 | 0.295 | 0.338 | 0.214 | |

Real 2026 scoring file (430 rows), scores with correctly built features vs the three features missing: Spearman 0.965, mean absolute probability difference 0.0098, maximum 0.118, **top-10% list overlap 88.4%** (about 1 in 8 high-risk names changes). The guard flags **430 of 430** delivered rows (coverage 0.0).

## Comparison and honest reading
* The hypothesis that silent imputation hurts **aggregate accuracy is not supported**: AUC is unchanged (the three features carry little signal in this model). The strict guard's higher AUC (0.718) is on a different, random 70% subset and is not an improvement; it is sampling variation, and the incomplete rows' AUC (0.693) differs by chance with n = 714.
* What the guard does show: (1) the delivered scoring file is entirely incomplete, a data-pipeline problem nobody was warned about; (2) individual rankings do shift near the cut-off.

## Trade-offs and decision
Withholding scores (strict mode) is **rejected**: it drops 30% coverage and would hide a third of the leavers. The adopted design is **flag, do not withhold**: score as usual, attach `data_status: review` and the missing-field list, and send flagged records to a human or to the monthly-feature builder first. This does not change model metrics (scenario A/C), only transparency.

## Limitations
Missingness was simulated at random; real missingness may be systematic (e.g. new joiners with no monthly history). The guard is not wired into the API (kept out of production code); integration is a small change to `api/v1/attrition.py` plus a response field, left for a future task. Thresholds (10%) are arbitrary.

## Lessons learned
Measure before assuming a risk is costly; the simple, honest finding was that the model is robust in aggregate but the data hand-off is broken. A guard's value can be transparency rather than accuracy, and a metric improvement on a subset is not evidence of improvement.
