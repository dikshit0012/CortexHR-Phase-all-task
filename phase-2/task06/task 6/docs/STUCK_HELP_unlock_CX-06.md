# CX-06 evaluation resource check

**Status: resolved in the current workspace.**

The current `02_CortexHR_AI_ML_Dataset.xlsx` contains the required `unlock_CX-06` worksheet. The Task 6 pipeline now selects that official worksheet first, uses it only after making predictions, and records the evaluation resource as `final` in `reports/task6_run_summary.json`.

## Verification

- `unlock_CX-06`: 100 rows, 5 columns, 8 error types.
- The worksheet's label rows match `payroll_anomaly_labels` exactly.
- All 100 `payroll_item_id` values match feature rows; there are no duplicate or unmatched label keys.
- The end-to-end run uses `unlock_CX-06` and emits no provisional-label warning.

No confirmation request to the internship team is needed unless they indicate that this workbook version is not authoritative. The historical fallback to `payroll_anomaly_labels` remains in the loader for older workbooks without the required sheet; such runs are clearly marked provisional.
