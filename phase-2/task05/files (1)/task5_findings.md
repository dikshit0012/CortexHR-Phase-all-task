# Task 5 – Spotting Unusual Attendance Without Labels

## 1. Objective
Build an attendance anomaly detector that does not use labels for training: Isolation Forest (3% contamination) plus a couple of simple business rules. The hidden labels are used only afterwards, to measure the finished detector and to study what it missed. The result is compared with the reference in `anomaly_baselines`.

## 2. Input data and features
Source: sheet `attendance_monthly_features` of `02_CortexHR_AI_ML_Dataset.xlsx` – 7,339 employee-months, 528 employees, Apr 2025 – Sep 2026, no missing values, no duplicate ids, no repeated employee-month.

Features given to the model (fixed before any evaluation):

| Feature | Why |
|---|---|
| attendance_rate | overall presence (counts half days) |
| unpaid_absence_days, longest_unpaid_streak_days | amount and concentration of unpaid absence |
| half_days_unpaid, paid_leave_days | separates excused from unexcused time away |
| late_arrivals | punctuality |
| avg_hours_worked, overtime_hours | working-time level |
| rate_change_vs_own_history | this month's rate minus the mean of the same employee's *earlier* months (current month excluded, first month = 0), so it needs no future data |

Left out: `scheduled_days`, `present_days` (mostly redundant with `attendance_rate`), `wfh_days` (depends on department policy), `job_level`, `department_id`, and all ids. The baseline used the raw 10 columns; the main differences here are the dropped redundant columns and the added history feature.

## 3. Preprocessing
Non-numeric text and ±inf become NaN, NaN is filled with the training median, and columns that are constant in the training data are dropped (none were). The imputer is fit on the same rows the forest is fit on, never on labels. The real file has no missing values, so this is protective; it is exercised by the unit tests.

## 4. Isolation Forest
`IsolationForest(n_estimators=200, contamination=0.03, random_state=42)`. sklearn's `-1/1` is converted to `if_flag` 1/0, and `if_anomaly_score = -decision_function` so that **higher = more abnormal**. Contamination sets the score cut-off at the 3rd percentile of the training scores, so about 3% of the *training* rows are flagged (221 of 7,339 here). It is not a guarantee: ties can move the count slightly, and new data scored with the same cut-off can be flagged at a different rate.

## 5. Business rules
| Rule | Checks | False positives to expect |
|---|---|---|
| `rule_unexcused_absence` | ≥ 5 consecutive unpaid absence days, **or** no presence in a month with scheduled days where paid leave doesn't cover the schedule | unrecorded approved unpaid leave, late leave entry |
| `rule_implausible_hours` | average hours per recorded day > 12 (with present_days > 0) | on-call/shift roles with a few very long days |

Months fully on paid leave, and months with no scheduled days, are deliberately not flagged (50 months in the data have zero presence; many are paid leave). Thresholds (5 days = one working week, 12 h) are from working-time logic, not tuned on results.

**Honesty note:** before designing the rules I had already looked at the labels sheet, whose `injection_note` column describes the injected anomalies (e.g. "10 consecutive unpaid days", "18.88 hours per day"). The rule thresholds are generic enough that I would have picked them anyway, but the rule-only precision below should be read as optimistic.

## 6. Combining
`combined_anomaly = if_flag OR rule_unexcused_absence OR rule_implausible_hours`. The output keeps separate columns for the IF flag, IF score, each rule, rule count and a plain-text `anomaly_reasons`. Adding rules can raise recall but can also add false positives.

## 7. How validation stayed label-free
1. `run_detector()` has no label argument; a unit test checks the signatures.
2. Predictions and `task5_flagged_records.csv` are written before the labels sheet is loaded.
3. Labels are used only for the metrics and the missed-anomaly analysis. No feature, threshold, rule or score was adjusted after seeing results.
4. The forest is fit on all 7,339 rows and scores the same rows (as the baseline does). This is an unsupervised, in-sample setup; it says nothing about a future month scored with a frozen model.

**Evaluator caveat:** `cortexhr_helpers.catch_rate()` and `unlock_CX-05` were not provided to me. `evaluation.py` contains a stand-in using precision = hits/flagged and recall = hits/injected. As a check, re-running the baseline recipe (raw 10 features, 3%) gives 0.398 / 0.587 against the published 0.403 / 0.593, so the definitions are consistent. Please re-run the headline numbers through your real helper.

## 8. Results (real, from `reports/task5_metrics.csv`)
| Detector | Precision | Recall | F1 | TP | FP | FN | Flagged | % of rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Published baseline | 0.403 | 0.593 | 0.480 | 89 | 132 | 61 | 221 | 3.01 |
| Baseline recipe re-run | 0.398 | 0.587 | 0.474 | 88 | 133 | 62 | 221 | 3.01 |
| Isolation Forest (our features) | 0.412 | 0.607 | 0.491 | 91 | 130 | 59 | 221 | 3.01 |
| Rule-based only | 0.918 | 0.600 | 0.726 | 90 | 8 | 60 | 98 | 1.34 |
| **Combined (IF OR rules)** | **0.488** | **0.827** | **0.614** | 124 | 130 | 26 | 254 | 3.46 |

Reading it:
- Isolation Forest alone is only marginally better than the baseline (+0.009 precision, +0.014 recall). Across seeds 0/1/7/42/123 it ranges 0.407–0.416 precision and 0.600–0.613 recall, so I would call that "about the same".
- The combined detector clearly improves on the baseline: recall 0.593 → 0.827 and precision 0.403 → 0.488, for 33 more flagged rows (254 vs 221). The gain comes mostly from the rules, which catch `IMPOSSIBLE_HOURS` that the forest almost never flags (2 of 35). Combined results are stable across seeds (precision 0.486–0.490, recall 0.827).
- The 8 rule-only "false positives" are months with 5–10 consecutive unpaid days that are not in the injected set. They look like genuine unpaid leave rather than errors, so HR should still look at them.
- Roughly 130 flagged rows are still false positives: the forest flags many unusual but unlabelled rows.

## 9. What was missed
Per type (from `task5_missed_anomalies.csv`):

| Type | Injected | Caught | Recall |
|---|---:|---:|---:|
| ABSENCE_STREAK | 40 | 40 | 1.00 |
| IMPOSSIBLE_HOURS | 35 | 35 | 1.00 |
| SUDDEN_DROP | 30 | 30 | 1.00 |
| ZERO_ATTENDANCE | 15 | 15 | 1.00 |
| **LATE_SPIKE** | 30 | 4 | **0.13** |

All 26 missed records are LATE_SPIKE:

| Missed anomaly type | Evidence from the data | Why the detector may have missed it | Possible improvement |
|---|---|---|---|
| LATE_SPIKE (26 of 30) | Missed rows average 16.1 late arrivals but attendance rate 0.98, 0.04 unpaid days, 8.55 h/day, history change +0.03: everything except `late_arrivals` looks normal. The 4 caught rows had lower rate (0.75) and 6.9 h/day. | One feature out of nine is extreme and the others are ordinary, so isolation depth is not short enough to clear the 3% cut-off; and no rule covers punctuality. | Add a punctuality rule or feature (late share of present days, change vs own history). Without looking at labels: only 30 of 7,339 rows have ≥ 10 late arrivals, so such a rule would be cheap to review. Not added here, since choosing it now would be tuning on the results. |

Other limits: the data has only monthly aggregates, so within-month patterns (e.g. Monday/Friday absences, moderate repeated absence below the streak threshold) cannot be seen at all, and I cannot say whether such anomalies exist in the labels because the labels contain only the five injected types.

## 10. HR ranking
Only flagged rows are ranked, no labels involved. Points: +1 if Isolation Forest flagged; +1 more if its score is in the top third of IF-flagged scores; +2 per rule triggered; +1 if the employee has ≥ 2 flagged months. **High ≥ 4, Medium 2–3, Low 1**; ties broken by IF score (higher = more abnormal). Points are ordinal, not probabilities.

Result: 47 High, 151 Medium, 56 Low. A post-hoc check (not used to build the tiers) shows 42 of the 47 High rows are injected anomalies (0.894), against 0.404 for Medium and 0.375 for Low, so the tiers do concentrate the real cases at the top. Limitation: the repeat-month point uses flags from the same run and is crude.

These are leads for a human to check, not evidence of misconduct. The output uses only pseudonymous ids (`employee_id`, `attendance_feature_id`) and no names or other personal data; keep the CSVs out of public repositories.

## 11. Limitations and next steps
- In-sample unsupervised evaluation on injected anomalies; real data will be messier.
- Stand-in evaluator (see §7); re-check with your `catch_rate()`.
- Rule-only precision is likely optimistic (§5).
- Single dataset; no time-based split.
- Next: punctuality feature/rule, per-department baselines, a frozen-model run on later months.

## 12. Reproduce
```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/run_attendance_anomaly_detection.py --xlsx /path/to/02_CortexHR_AI_ML_Dataset.xlsx
```
