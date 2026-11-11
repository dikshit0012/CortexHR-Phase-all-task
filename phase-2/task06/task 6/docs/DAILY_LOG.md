# Daily Log
Copy one block per working day.

**Date:** 2026-10-26 (Monday)
**Done today:**
- Built the Task 6 payroll outlier detector: 9 ratio features, 10 business rules, Isolation Forest and the combined "rules OR model" detector, with 90 tests on synthetic data.
- Scored all 8 error types (precision, recall, F1, 95% intervals) against `payroll_anomaly_labels` and compared with the `anomaly_baselines` row.
- A code review found that results depended on the order of the input rows (duplicate rule, model fit, "previous month" lookup). Fixed it, froze the original results in `reports/original_experiment/`, and documented the before/after numbers.
- Reviewed `rule_salary_drop` (13 false alarms, 0 hits). Left it unchanged and documented the suppression idea as an unadopted hypothesis.
- Fixed two Windows risks (an Excel file handle left open, UTF-8 not stated), wrote the architecture diagram, release notes and the other team templates.
**Doing next:**
- Run the four commands on my Windows machine and tick the "Tests pass" box in the pull request.
- Ask the internship team whether `payroll_anomaly_labels` is acceptable in place of `unlock_CX-06` (see STUCK_HELP_unlock_CX-06.md).
- Record the demo video, open the pull request.
**Blocked by:** The `unlock_CX-06` evaluation sheet is not in `02_CortexHR_AI_ML_Dataset.xlsx` (sheets containing "label": attendance_anomaly_labels, payroll_anomaly_labels, resume_job_labels). All results are marked provisional until the team answers.
**One thing I learned:** (draft - rewrite in your own words) A passing test suite can hide a real bug: my first 57 tests passed, but shuffling the input rows changed the results, because the duplicate rule flagged "the later row in the file" and the model was fitted in file order. A simple row-shuffle test found it.

**Follow-up audit (same date):**
- Confirmed the current workbook copy does contain `unlock_CX-06`; its 100 labels match `payroll_anomaly_labels` exactly. Reran the pipeline against the official sheet and regenerated the reports.
- Changed the default role-pay baseline to use the previous calendar month's role median, eliminating future-month influence. Added key/date validation and regression tests.
- Windows Task 6 suite: 95 passed; end-to-end run and independent report check passed.
- Final privacy hardening replaced the public demo masking key with a random per-process key by default; configurable secrets support stable pseudonyms when needed.
- Final Windows suite after the privacy change: 96 passed. Re-ran the real-data pipeline, report reconciliation, and frozen original-metric comparison successfully.
- Task 5 tests and Git status could not be checked because neither `task 5/` nor `.git/` exists in this workspace.
- **Current blocker:** staging/history review and Task 5 regression require the complete Git checkout. The prior `unlock_CX-06` blocker is resolved for this workbook.
