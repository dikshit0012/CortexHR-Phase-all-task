# Decision log — Task 1 audit

Row counts are on the raw 2,389-row table (before de-duplication). Each row is a rule the Task 2 pipeline must implement (D03 and D24 are on HOLD pending the Python team). Rows = how many records are affected.

| ID | Area | Issue | Rows | Decision | Why | Task 2 rule |
|---|---|---|---|---|---|---|
| D01 | Duplicates | 6 exact duplicate rows (employee appears twice) | 6 | DROP second copy | Duplicates inflate scores when copies fall in train and test | dedupe on all columns except row_id; GroupKFold by employee_id |
| D02 | Consistency | 13 city labels spelled differently (Bangalore, BENGALURU, Madras, Delhi-NCR ...) | 13 | FIX with lookup table | Same city must be one category | normalise_location() in src/cortexhr/audit.py |
| D03 | Invalid | Negative tenure (obs. date before join date): EMP-0248, 0280, 0568, 0596 | 4 | HOLD - open item O1. Raw value left unchanged; add tenure_was_invalid flag; until the team answers, EXCLUDE these 4 rows from model training (documented, not imputed) | True join dates are unknown; any fix would be a guess | flag tenure<0; drop flagged rows from training until O1 is closed, then apply the team's answer |
| D04 | Invalid | Attendance rate above 100% (1.12) | 2 | CAP at 1.0 + FLAG | Cannot exceed 100% | clip(0,1) |
| D05 | Invalid | wfh_share_3m outside 0-1 | 213 | CLIP to 0-1 | Small noise (|x|<0.1); a share must lie in 0-1 | clip(0,1) |
| D06 | Invalid | Total experience below tenure (137 rows) or negative (10 rows), all in historical rows | 137 | FLOOR at tenure in years | Total experience includes time at this company | total_exp=max(total_exp, tenure_months/12) |
| D07 | Outlier | Salary typos: 3 values 10x too high (EMP-0034, 0433, 0601) | 3 | CORRECT (divide by 10) | pay/comp_ratio disagrees with role mid-point by ~11x; pay/10 falls inside the range for the level (inference, not confirmed) | if pay/comp_ratio > 3x role median: pay/=10 |
| D08 | Outlier | Genuine high salaries (IQR flags 93, z-score 26) | 93 | KEEP | Senior staff are paid more; this is real | log1p(annual_fixed_pay) for linear models |
| D09 | Outlier | Daily hours 1.8 / attendance 20% (EMP-0313) | 1 | KEEP + FLAG for HR review | Two columns agree, so looks like a real long absence | RobustScaler / tree models |
| D10 | Outlier | Overtime heavy tail (81% zeros, max 50 h); tenure up to 164 m |  | KEEP | Plausible values; IQR over-flags zero-heavy data | log1p(overtime) |
| D11 | Missing | months_since_last_promotion 61% blank = never promoted | 1459 | KEEP; FILL with tenure_months; add never_promoted flag | Blank carries meaning; dropping loses signal | fill=tenure_months; flag |
| D12 | Missing | last_raise_pct 8.5% blank = never had a raise | 202 | KEEP; FILL 0; add never_raised flag | Structural blank | fill=0; flag |
| D13 | Missing | last_performance_rating 7.6% blank = joined <12 months ago | 182 | KEEP; FILL median; add no_rating_yet flag | Structural blank | median from TRAIN only; flag |
| D14 | Missing | last_engagement_score 22% blank = no survey response | 528 | KEEP; FILL median; add engagement_missing flag | Non-response may carry signal (16.7% vs 14.0% attrition, not significant) | median by department from TRAIN only; flag |
| D15 | Missing | commute_distance_km 2.3% blank | 55 | FILL median by city; add flag | Random gap, small | median by city from TRAIN only; flag |
| D16 | Missing | highest_education 0.6% blank | 15 | FILL 'Unknown' | Random gap, categorical | constant 'Unknown' |
| D17 | Imbalance | 14.6% left, 85.4% stayed (5.8 : 1) | 349 | KEEP the data as is; no copying rows | Real-world rate; oversampling before splitting leaks | StratifiedGroupKFold; class_weight='balanced'; metrics = ROC-AUC, PR-AUC, recall@top-10% |
| D18 | Leakage | exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount (post-outcome) |  | DROP before any modelling | Exist only after the outcome; give AUC 1.0 | assert none of LEAKAGE_COLUMNS in feature matrix |
| D19 | Leakage | voluntary_left_within_12m (sub-label of the target) |  | DROP | Derived from the answer | same assertion |
| D20 | Leakage | Survey, appraisal and attendance sheets include data after 30 Sep 2025 |  | CUT OFF at observation_date when building features | Future information would leak in | filter date <= observation_date |
| D21 | Non-features | row_id, employee_id, source, observation_date |  | EXCLUDE from features (employee_id for grouping; observation_date for time split) | Identifiers/provenance, not employee traits | NON_FEATURE_COLUMNS |
| D22 | Fairness | gender and age are held in a separate sheet |  | KEEP OUT of the feature set; use only to test fairness (Task 17) | Avoid direct discrimination; still need them to test for bias | never join protected_attributes into X |
| D24 | Leakage (unconfirmed) | dept_attrition_rate_prior_12m: cut-off date not confirmed |  | HOLD - open item O2. Keep as a CANDIDATE feature; train every model with and without it and report both until confirmed | If it used exits after the observation date it would leak the answer; the evidence so far is only consistent with a backward-looking calculation | two feature sets (with/without); drop it if the team says it uses post-observation exits |
| D23 | Process | Imputation, scaling and encoding must not see the test data |  | FIT on training fold only (sklearn Pipeline) | Otherwise test information leaks in | all transforms inside Pipeline/ColumnTransformer |
