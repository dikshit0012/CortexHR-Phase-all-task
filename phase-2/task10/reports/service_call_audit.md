# Task 10: service-call audit (official dialogues)

Generated 2026-10-30 03:14 UTC by scripts/run_evidence.py. Reference date for 'last month' = 2026-04-15. mock_data is SYNTHETIC (the real mock_data CSVs were not supplied); pay values are sentinels and are never printed.

Every HTTP request made while running the 12 dialogues, with the identity header actually sent.

| Dialogue | Turn | Intent | HTTP calls observed | Count |
|---|---|---|---|---|
| DLG-01 | 1 | `leave_balance` | GET /api/v1/leave/balance?employee=me | 1 |
| DLG-01 | 2 | `leave_balance` | GET /api/v1/leave/balance?employee=me | 1 |
| DLG-01 | 3 | `thanks` | none | 0 |
| DLG-02 | 1 | `payslip_status` | GET /api/v1/payslips/me/2026-03 | 1 |
| DLG-02 | 2 | `my_manager` | GET /api/v1/employees/me | 1 |
| DLG-03 | 1 | `apply_leave_howto` | none | 0 |
| DLG-03 | 2 | `hr_policy_leave` | none | 0 |
| DLG-03 | 3 | `goodbye` | none | 0 |
| DLG-04 | 1 | `attendance_summary` | GET /api/v1/attendance/summary?employee=me&month=2026-03 | 1 |
| DLG-04 | 2 | `None` | none | 0 |
| DLG-05 | 1 | `salary_of_other` | none | 0 |
| DLG-05 | 2 | `payslip_status` | GET /api/v1/payslips/me/2026-03 | 1 |
| DLG-06 | 1 | `greet` | none | 0 |
| DLG-06 | 2 | `chitchat` | none | 0 |
| DLG-06 | 3 | `wfh_policy` | none | 0 |
| DLG-07 | 1 | `attrition_risk_query` | GET /ai/v1/attrition/risk | 1 |
| DLG-07 | 2 | `attrition_risk_query` | GET /ai/v1/attrition/risk | 1 |
| DLG-07 | 3 | `attrition_risk_query` | GET /ai/v1/attrition/risk | 1 |
| DLG-08 | 1 | `payroll_flags` | GET /ai/v1/anomaly/payroll | 1 |
| DLG-08 | 2 | `payroll_flags` | GET /ai/v1/anomaly/payroll | 1 |
| DLG-09 | 1 | `salary_band_query` | POST /ai/v1/salary-band | 1 |
| DLG-09 | 2 | `salary_band_query` | POST /ai/v1/salary-band | 1 |
| DLG-10 | 1 | `headcount_forecast` | GET /ai/v1/forecast/headcount | 1 |
| DLG-10 | 2 | `headcount_forecast` | GET /ai/v1/forecast/headcount | 1 |
| DLG-11 | 1 | `resume_score_request` | POST /ai/v1/resume/score | 1 |
| DLG-11 | 2 | `thanks` | none | 0 |
| DLG-12 | 1 | `out_of_scope` | none | 0 |
| DLG-12 | 2 | `help` | none | 0 |

Identities used: employee dialogues -> `EMP-1001` only; HR dialogues -> `EMP-2001` only (the user id never comes from message text).