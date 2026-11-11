# Task 10: official dialogue report (12 dialogues, 28 turns)

Generated 2026-10-30 03:14 UTC by scripts/run_evidence.py. Reference date for 'last month' = 2026-04-15. mock_data is SYNTHETIC (the real mock_data CSVs were not supplied); pay values are sentinels and are never printed.

Pass rule: every turn of a dialogue must satisfy every check (intent == official `expected_intent`, authorization as configured, service call exactly as configured, follow-up/slot expectations).

| Dialogue | Role | Turns | Result |
|---|---|---|---|
| DLG-01 | employee | 3 | **PASS** |
| DLG-02 | employee | 2 | **PASS** |
| DLG-03 | employee | 3 | **PASS** |
| DLG-04 | employee | 2 | **FAIL** |
| DLG-05 | employee | 2 | **PASS** |
| DLG-06 | employee | 3 | **PASS** |
| DLG-07 | hr | 3 | **PASS** |
| DLG-08 | hr | 2 | **PASS** |
| DLG-09 | hr | 2 | **PASS** |
| DLG-10 | hr | 2 | **PASS** |
| DLG-11 | hr | 2 | **PASS** |
| DLG-12 | employee | 2 | **PASS** |

**11 of 12 dialogues PASS, 1 FAIL.**

## DLG-01: PASS

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | how many casual leaves do I have left | `leave_balance` | `leave_balance` (0.6767) | answered / ok | - | GET /api/v1/leave/balance?employee=me | yes | PASS |
| 2 | and sick leave? | `leave_balance` | `leave_balance` (0.1966) | answered / ok | - | GET /api/v1/leave/balance?employee=me | yes | PASS |
| 3 | thanks | `thanks` | `thanks` (0.9956) | static / ok | - | none | yes | PASS |

## DLG-02: PASS

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | is my March payslip ready | `payslip_status` | `payslip_status` (0.9458) | answered / ok | {'month': '2026-03'} | GET /api/v1/payslips/me/2026-03 | yes | PASS |
| 2 | who is my manager | `my_manager` | `my_manager` (0.9741) | answered / ok | - | GET /api/v1/employees/me | yes | PASS |

## DLG-03: PASS

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | how do I apply for leave | `apply_leave_howto` | `apply_leave_howto` (0.982) | static / ok | - | none | yes | PASS |
| 2 | what is the leave policy | `hr_policy_leave` | `hr_policy_leave` (0.9396) | static / ok | - | none | yes | PASS |
| 3 | ok bye | `goodbye` | `goodbye` (0.9812) | static / ok | - | none | yes | PASS |

## DLG-04: FAIL

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | show my attendance for last month | `attendance_summary` | `attendance_summary` (0.7642) | answered / ok | {'month': '2026-03'} | GET /api/v1/attendance/summary?employee=me&month=2026-03 | yes | PASS |
| 2 | any unusual days? | `unusual_attendance` | `None` (0.5874) | fallback / n/a | {'month': '2026-03'} | none | NO | FAIL |

**Failed checks, turn 2:** intent == official expected_intent; authorization as configured; exactly one service call, matching the configured endpoint; answered (template filled, no failure)
Actual reply: "I'm not sure I understood your question. Please rephrase it, or contact HR for assistance."  (classifier confidence 0.5874 vs threshold 0.64; the Task 9 model's top intent was the expected one.)

## DLG-05: PASS

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | what is Priya Nair salary | `salary_of_other` | `salary_of_other` (0.7321) | refused / refuse_always | - | none | yes | PASS |
| 2 | show my payslip for last month | `payslip_status` | `payslip_status` (0.9689) | answered / ok | {'month': '2026-03'} | GET /api/v1/payslips/me/2026-03 | yes | PASS |

## DLG-06: PASS

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | hi | `greet` | `greet` (0.0634) | static / ok | - | none | yes | PASS |
| 2 | what can you do | `chitchat` | `chitchat` (0.919) | static / ok | - | none | yes | PASS |
| 3 | work from home policy | `wfh_policy` | `wfh_policy` (0.9727) | static / ok | - | none | yes | PASS |

## DLG-07: PASS

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | who is at risk of leaving in Sales | `attrition_risk_query` | `attrition_risk_query` (0.954) | answered / ok | {'department': 'Sales'} | GET /ai/v1/attrition/risk | yes | PASS |
| 2 | and in Engineering? | `attrition_risk_query` | `attrition_risk_query` (0.3006) | answered / ok | {'department': 'Engineering'} | GET /ai/v1/attrition/risk | yes | PASS |
| 3 | why is it high | `attrition_risk_query` | `attrition_risk_query` (0.6529) | answered / ok | {'department': 'Engineering'} | GET /ai/v1/attrition/risk | yes | PASS |

## DLG-08: PASS

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | any unusual payroll this month | `payroll_flags` | `payroll_flags` (0.979) | answered / ok | {'month': '2026-04'} | GET /ai/v1/anomaly/payroll | yes | PASS |
| 2 | show details | `payroll_flags` | `payroll_flags` (0.3805) | answered / ok | {'month': '2026-04'} | GET /ai/v1/anomaly/payroll | yes | PASS |

## DLG-09: PASS

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | suggest a salary band for Data Analyst | `salary_band_query` | `salary_band_query` (0.9086) | answered / ok | {'role': 'Data Analyst'} | POST /ai/v1/salary-band | yes | PASS |
| 2 | what about Software Engineer | `salary_band_query` | `salary_band_query` (0.2266) | answered / ok | {'role': 'Software Engineer'} | POST /ai/v1/salary-band | yes | PASS |

## DLG-10: PASS

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | forecast headcount for next quarter | `headcount_forecast` | `headcount_forecast` (0.9799) | answered / ok | - | GET /ai/v1/forecast/headcount | yes | PASS |
| 2 | for Support only | `headcount_forecast` | `headcount_forecast` (0.2337) | answered / ok | - | GET /ai/v1/forecast/headcount | yes | PASS |

## DLG-11: PASS

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | score this resume for Data Analyst | `resume_score_request` | `resume_score_request` (0.981) | answered / ok | {'role': 'Data Analyst'} | POST /ai/v1/resume/score | yes | PASS |
| 2 | thanks | `thanks` | `thanks` (0.9956) | static / ok | - | none | yes | PASS |

## DLG-12: PASS

| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | tell me a joke | `out_of_scope` | `out_of_scope` (0.8944) | static / ok | - | none | yes | PASS |
| 2 | help | `help` | `help` (0.9738) | static / ok | - | none | yes | PASS |
