# Release notes - Task 6 Payroll Outlier Detector

## v1.2 - causal role reference and official evaluation (current)
- The default role-pay reference is now the same role's median fixed gross from the **previous calendar month**. Future payslips no longer influence earlier feature values; missing prior months have no reference.
- Payroll key fields, month-start dates, and `run_id`/month consistency are validated before feature engineering.
- Masking now uses a random per-process key by default rather than a public demo key, and emits a 64-bit HMAC reference; a secret `CORTEXHR_ID_SALT` can be configured when cross-run stable references are needed.
- The current workbook includes the required `unlock_CX-06` sheet. The main reports were regenerated against it; `payroll_anomaly_labels` is confirmed identical and remains only a compatibility fallback.
- Windows verification: 96 Task 6 tests passed; the real-data run and independent report reconciliation succeeded. Task 5 tests were unavailable because this workspace contains no Task 5 directory.
- The first experiment's frozen files are preserved. `--original-experiment` uses an explicit full-window `RoleReference` to retain the historical comparison; only the corrected default uses the causal previous-month reference.

## v1.1 - corrected, order-independent (previous default)
**Why:** a code review showed the first results depended on the order of the input rows.
- Duplicate-payment rule: now flags every payslip in a same-employee, same-month, same-amount group (the data has no payment reference to say which copy is the extra one). Options `later_by_item_id` and the original `file_order` remain available.
- Isolation Forest: rows are put in a canonical order before fitting, so the flags no longer change when the file is shuffled.
- History features: "previous month" now means the previous *calendar* month; a missing month gives no history.
- New: 95% confidence intervals, original-vs-corrected-vs-exploratory table, `rule_salary_drop` review table, experimental tiered-alert view, baseline-replica seed check, independent report-reconciliation script, Windows fixes (closed Excel handle, explicit UTF-8), architecture diagram, project documents.
- Tests at the time: 90 (was 57). At that time the evaluation was provisional because the workbook copy then available lacked `unlock_CX-06`; the current workbook now contains the official sheet (see v1.2).

| Detector | v1.0 original (flagged / precision / recall) | v1.1 corrected |
|---|---|---|
| Rules only | 105 / 0.876 / 0.920 | 120 / 0.767 / 0.920 |
| Isolation Forest only | 184 / 0.304 / 0.560 | 184 / 0.304 / 0.560 |
| Combined | 238 / 0.408 / 0.970 | 253 / 0.383 / 0.970 |

Rules precision fell because the legitimate original of each duplicate pair is now flagged too (the labels mark only the injected copy).

## v1.0 - original experiment (frozen)
- Files in `phase-2/task06/task 6/reports/original_experiment/`; re-create with `python "phase-2/task06/task 6/scripts/run_task6.py" --original-experiment`.
- 9 ratio features, 10 rules, Isolation Forest (seed 42, contamination 2.5%), combined detector, per-error-type evaluation, baseline comparison.
- Known weaknesses found later: results depended on row order (duplicate rule, model fit); "previous month" used the previous row.

## Not changed
- `rule_salary_drop` (13 false alarms, 0 hits on the substitute labels) is kept as specified. An opt-in suppression variant exists but is **not adopted**, because it could hide a genuine stale-salary error.
- Task 5 is untouched.
