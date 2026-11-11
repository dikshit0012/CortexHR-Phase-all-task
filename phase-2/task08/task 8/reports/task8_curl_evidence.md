# Windows PowerShell HTTP evidence

Captured from a local Uvicorn process on Windows using the pinned Task 8 dependency set. Requests and IDs are fictional examples.

## health

```text
{"status":"ok","service":"cortexhr-anomaly","models_loaded":true}
HTTP 200
```
## models

```text
{"max_batch_size":100,"detectors":{"attendance":{"artifact_version":"attendance-anomaly-v1","response_model_version":"iforest+rules-attendance-task5","created_at":"2026-10-28 10:59:19","reference_rows":7339,"model_features":["attendance_rate","unpaid_absence_days","longest_unpaid_streak_days","half_days_unpaid","paid_leave_days","late_arrivals","avg_hours_worked","overtime_hours","rate_change_vs_own_history"],"sklearn_version":"1.9.1","dependency_versions":{"python":"3.14.6","numpy":"2.5.4","pandas":"3.1.0","scipy":"1.18.1","sklearn":"1.9.1","joblib":"1.6.0","openpyxl":"3.1.5"},"score_scale":{"min":-0.24875952240657284,"max":0.20868581880348291}},"payroll":{"artifact_version":"payroll-anomaly-v1","response_model_version":"iforest+rules-payroll-task6","created_at":"2026-10-28 10:59:19","reference_rows":7355,"model_features":["gross_to_fixed_gross","deduction_ratio","net_pay_ratio","gross_vs_expected","fixed_gross_vs_role_median","fixed_gross_change_vs_prev","overtime_hours_paid","overtime_pay_share","pf_vs_prev_month"],"sklearn_version":"1.9.1","dependency_versions":{"python":"3.14.6","numpy":"2.5.4","pandas":"3.1.0","scipy":"1.18.1","sklearn":"1.9.1","joblib":"1.6.0","openpyxl":"3.1.5"},"score_scale":{"min":-0.21614528298243668,"max":0.1689582416369514}}}}
HTTP 200
```
## attendance single

```text
{"result_id":"AN-0001","source_dataset":"attendance_monthly_features","record_id":"AF-EX-0001","employee_id":"EMP-0002","period":"2026-10-01","anomaly_score":1.0,"flagged":"Y","reason_code":"NO_ATTENDANCE","reason_text":"No attendance was recorded for the whole month although 21 working days were scheduled and paid leave does not cover them. Please check whether the employee was absent, on leave that was not recorded, or missing from the attendance system.","money_at_risk":null,"model_version":"iforest+rules-attendance-task5","scored_at":"2026-10-28 11:04:12"}
HTTP 200
```
## payroll single

```text
{"result_id":"AN-0001","source_dataset":"payroll_monthly_features","record_id":"PI-EX-0001","employee_id":"EMP-0002","period":"2026-10-01","anomaly_score":0.667,"flagged":"Y","reason_code":"GROSS_ABOVE_NORM","reason_text":"The gross pay (1,600,000.00) is higher than the pay components on this payslip add up to (160,000.00). About 1,440,000.00 may be affected. Please check the fixed pay, overtime and bonus amounts.","money_at_risk":1440000.0,"model_version":"iforest+rules-payroll-task6","scored_at":"2026-10-28 11:04:15"}
HTTP 200
```
## attendance batch

```text
{"detector":"attendance","count":3,"results":[{"result_id":"AN-0003","source_dataset":"attendance_monthly_features","record_id":"AF-EX-0101","employee_id":"EMP-0002","period":"2026-10-01","anomaly_score":0.116,"flagged":"N","reason_code":null,"reason_text":null,"money_at_risk":null,"model_version":"iforest+rules-attendance-task5","scored_at":"2026-10-28 11:04:16"},{"result_id":"AN-0001","source_dataset":"attendance_monthly_features","record_id":"AF-EX-0102","employee_id":"EMP-0004","period":"2026-10-01","anomaly_score":1.0,"flagged":"Y","reason_code":"NO_ATTENDANCE","reason_text":"No attendance was recorded for the whole month although 21 working days were scheduled and paid leave does not cover them. Please check whether the employee was absent, on leave that was not recorded, or missing from the attendance system.","money_at_risk":null,"model_version":"iforest+rules-attendance-task5","scored_at":"2026-10-28 11:04:16"},{"result_id":"AN-0002","source_dataset":"attendance_monthly_features","record_id":"AF-EX-0103","employee_id":"EMP-0006","period":"2026-10-01","anomaly_score":0.14,"flagged":"N","reason_code":null,"reason_text":null,"money_at_risk":null,"model_version":"iforest+rules-attendance-task5","scored_at":"2026-10-28 11:04:16"}]}
HTTP 200
```
## payroll batch

```text
{"detector":"payroll","count":2,"results":[{"result_id":"AN-0002","source_dataset":"payroll_monthly_features","record_id":"PI-EX-0101","employee_id":"EMP-0002","period":"2026-10-01","anomaly_score":0.09,"flagged":"N","reason_code":null,"reason_text":null,"money_at_risk":null,"model_version":"iforest+rules-payroll-task6","scored_at":"2026-10-28 11:04:19"},{"result_id":"AN-0001","source_dataset":"payroll_monthly_features","record_id":"PI-EX-0102","employee_id":"EMP-0004","period":"2026-10-01","anomaly_score":0.696,"flagged":"Y","reason_code":"GROSS_ABOVE_NORM","reason_text":"The gross pay (1,000,000.00) is higher than the pay components on this payslip add up to (100,000.00). About 900,000.00 may be affected. Please check the fixed pay, overtime and bonus amounts.","money_at_risk":900000.0,"model_version":"iforest+rules-payroll-task6","scored_at":"2026-10-28 11:04:19"}]}
HTTP 200
```
## invalid attendance

```text
{"detail":[{"loc":["body","present_days"],"type":"greater_than_equal","msg":"Input should be greater than or equal to 0"},{"loc":["body","attendance_rate"],"type":"less_than_equal","msg":"Input should be less than or equal to 1"}]}
HTTP 422
```
## invalid payroll

```text
{"detail":[{"loc":["body","overtime_hours_paid"],"type":"greater_than_equal","msg":"Input should be greater than or equal to 0"}]}
HTTP 422
```
## empty payroll batch

```text
{"detail":[{"loc":["body","records"],"type":"too_short","msg":"List should have at least 1 item after validation, not 0"}]}
HTTP 422
```
## unknown detector

```text
{"detail":"Not Found"}
HTTP 404
```

## Invoke-RestMethod
HTTP 200; record_id=AF-EX-0001
