# Task 5 - Spotting Unusual Attendance Without Labels

## 1. Objective
Build an attendance anomaly detector that does **not** use labels for training: Isolation Forest (3% contamination) plus simple business rules. Hidden labels are used **only afterwards** to measure the finished detector and study what it missed. The result is compared with the reference in `anomaly_baselines`.

## 2. Input Data and Features

**Source:** sheet `attendance_monthly_features` from `02_CortexHR_AI_ML_Dataset.xlsx`
- **7,339** employee-months
- **528** employees
- **18** months span (Apr 2025 - Sep 2026)
- No missing values, no duplicate IDs

### Features Given to the Model

| Feature | Why |
|---|---|
| `attendance_rate` | Overall presence (counts half days) |
| `unpaid_absence_days` | Amount of unpaid absence |
| `longest_unpaid_streak_days` | Concentration of unpaid absence |
| `half_days_unpaid` | Separates excused from unexcused |
| `paid_leave_days` | Legitimate time away |
| `late_arrivals` | Punctuality |
| `avg_hours_worked` | Working-time level |
| `overtime_hours` | Extra work |
| `rate_change_vs_own_history` | **Engineered:** this month's rate minus the mean of the same employee's *earlier* months (current month excluded, first month = 0) - no future data leakage |

**Left out:** `scheduled_days`, `present_days` (redundant with `attendance_rate`), `wfh_days` (department policy), `job_level`, `department_id` (identifiers).

## 3. Preprocessing
- Non-numeric text and infinities become NaN
- NaN is filled with the training median (`SimpleImputer`)
- Columns constant in training data are dropped (none were)
- Imputer is fit on the same rows as the forest, **never on labels**

## 4. Isolation Forest

```
IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
```

- sklearn's `-1/1` output is converted to `if_flag` (1=anomaly, 0=normal)
- `if_anomaly_score = -decision_function` so **higher = more abnormal**
- ~3% contamination flagged **221** of 7,339 rows

## 5. Business Rules

| Rule | What It Checks | Likely False Positives |
|---|---|---|
| `rule_unexcused_absence` | >= 5 consecutive unpaid absence days, OR no presence in a month with scheduled days where paid leave doesn't cover the schedule | Unrecorded approved unpaid leave, late leave entry |
| `rule_implausible_hours` | Average hours per day > 12 (with present_days > 0) | On-call/shift roles with a few very long days |

**Key design:** Months fully on paid leave and months with no scheduled days are deliberately **not** flagged. Thresholds (5 days = one working week, 12 hours) come from working-time logic, not tuned on results.

## 6. Combining Model + Rules

```
combined_anomaly = IF_flag OR rule_unexcused_absence OR rule_implausible_hours
```

Output keeps separate columns for each signal source plus a plain-text `anomaly_reasons`.

## 7. Label-Free Training Guarantee

1. `run_detector()` has **no label argument** - unit test verifies this
2. Predictions and `task5_flagged_records.csv` are saved **BEFORE** labels are loaded
3. Labels are used only for metrics and missed-anomaly analysis
4. No feature, threshold, rule, or score was adjusted after seeing results
5. **14/14 unit tests pass**, including `test_no_label_arguments_anywhere`

---

## 8. Results - Precision and Recall

| Detector | Precision | Recall | F1 | TP | FP | FN | Flagged | % of rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Published baseline | 0.403 | 0.593 | 0.480 | 89 | 132 | 61 | 221 | 3.01% |
| Baseline recipe re-run | 0.398 | 0.587 | 0.474 | 88 | 133 | 62 | 221 | 3.01% |
| Isolation Forest (our features) | 0.412 | 0.607 | 0.491 | 91 | 130 | 59 | 221 | 3.01% |
| Rule-based only | **0.918** | 0.600 | 0.726 | 90 | 8 | 60 | 98 | 1.34% |
| **Combined (IF OR rules)** | **0.488** | **0.827** | **0.614** | **124** | 130 | **26** | 254 | 3.46% |

### Key Takeaways
- **IF alone vs baseline:** Marginal improvement (+0.009 precision, +0.014 recall). The history feature helps slightly.
- **Rules alone:** Extremely precise (0.918!) but limited recall (0.600) - only catches absence/hours types.
- **Combined detector clearly beats the baseline:** Recall 0.593 -> **0.827** (+39%), Precision 0.403 -> **0.488** (+21%), for just 33 more flagged rows.
- The gain comes mostly from rules catching `IMPOSSIBLE_HOURS` that the forest rarely flags (2 of 35).

---

## 9. Missed Anomaly Analysis

| Anomaly Type | Injected | Caught | Missed | Recall | Caught by IF | Caught by Rules |
|---|---:|---:|---:|---:|---:|---:|
| ABSENCE_STREAK | 40 | 40 | 0 | **1.000** | 40 | 40 |
| IMPOSSIBLE_HOURS | 35 | 35 | 0 | **1.000** | 2 | 35 |
| SUDDEN_DROP | 30 | 30 | 0 | **1.000** | 30 | 0 |
| ZERO_ATTENDANCE | 15 | 15 | 0 | **1.000** | 15 | 15 |
| **LATE_SPIKE** | **30** | **4** | **26** | **0.133** | 4 | 0 |

### All 26 Missed Records Are LATE_SPIKE

| What We See in Missed Records | Why the Detector Misses It | Possible Fix |
|---|---|---|
| Average 16.1 late arrivals, but attendance_rate 0.98, unpaid_absence_days 0.04, avg_hours 8.55, history change +0.03. Everything except `late_arrivals` looks completely normal. | Only 1 feature out of 9 is extreme while the other 8 are ordinary. Isolation depth is not short enough to clear the 3% cut-off. No business rule covers punctuality. | Add a punctuality rule (e.g., late_arrivals >= 10) or a `late_share` feature. Only 30 of 7,339 rows have >= 10 lates, so such a rule would be cheap to review. **Not added here** to avoid tuning on results. |
| The 4 caught LATE_SPIKE rows had lower attendance_rate (0.75) and avg_hours (6.86), making them unusual on multiple dimensions simultaneously. | When a LATE_SPIKE also has low attendance, the forest catches it. Pure late-spike-only records blend in. | - |

### Why LATE_SPIKE Is Hard
The fundamental challenge: Isolation Forest works by how easy a point is to **isolate** across all feature dimensions. When only 1 of 9 features is abnormal and the rest are perfectly normal, the total isolation score may not cross the threshold. This is a known limitation of tree-based anomaly detectors for **single-feature anomalies in high-dimensional spaces**.

---

## 10. HR Ranking System

Only flagged rows are ranked (no labels involved):

| Points | Source |
|---|---|
| +1 | Isolation Forest flagged |
| +1 | IF score in top third of IF-flagged scores |
| +2 | Per business rule triggered |
| +1 | Employee has >= 2 flagged months (repeat pattern) |

| Priority | Points | Flagged | True Anomalies | Precision |
|---|---:|---:|---:|---:|
| **High** | >= 4 | 47 | 42 | **0.894** |
| Medium | 2-3 | 151 | 61 | 0.404 |
| Low | 1 | 56 | 21 | 0.375 |

**Result:** The High tier concentrates 89.4% true positives, meaning an HR reviewer checking High-priority flags first would find a real anomaly 9 out of 10 times.

### Answering: "If a detector flags 1,000 rows a month, who has time to check them?"
With 254 flags over 18 months (~14/month), the volume is manageable. But at scale:
1. **Prioritize:** Use the points-based tiers. Check High first (47 rows = ~2.6/month, 89% hit rate).
2. **Batch by type:** Group by `anomaly_reasons` so one reviewer handles all "impossible hours" at once.
3. **Auto-resolve:** Some patterns (e.g., clock-in/out errors creating impossible hours) can be auto-flagged for payroll correction without human review.
4. **Feedback loop:** Track which flags HR dismisses to adjust thresholds over time.

---

## 11. Completion Checklist

- [x] **Precision and recall reported** - Combined: P=0.488, R=0.827, F1=0.614
- [x] **Note on hard anomaly types** - LATE_SPIKE (recall 0.133): only 1 feature extreme, rest normal
- [x] **Never trained on labels** - Labels loaded only after predictions; unit test `test_no_label_arguments_anywhere` passes
- [x] **Rules and model both explained** - IF for multivariate patterns; rules for domain-specific thresholds
- [x] **All 14 unit tests pass**
- [x] **Pipeline runs end-to-end** on real data

## 12. Project Structure

```
phase-2/task05/
  src/cortexhr_task5/
    __init__.py
    attendance_anomaly_detector.py   # detector + rules (label-free)
    evaluation.py                     # catch_rate(), label loading
  scripts/
    run_task5.py                      # end-to-end pipeline
    run_attendance_anomaly_detection.py  # original runner (argparse)
  tests/
    test_attendance_anomaly_detector.py  # 14 unit tests
  reports/
    task5_metrics.csv                 # precision/recall comparison table
    task5_flagged_records.csv         # all flagged rows with reasons
    task5_missed_anomalies.csv        # per-type breakdown
    task5_missed_record_ids.csv       # individual missed record details
    task5_priority_tiers.csv          # review tier analysis
  requirements.txt
```

## 13. How to Reproduce

```bash
pip install -r requirements.txt
python -m pytest tests/ -v                # 14/14 pass
python scripts/run_task5.py               # full pipeline
```

## 14. Reference Links
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [pandas](https://github.com/pandas-dev/pandas)
- [PyOD (anomaly detection toolkit)](https://github.com/yzhao062/pyod)
