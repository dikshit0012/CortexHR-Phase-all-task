# Data dictionary — `attrition_dataset`

Written for Task 1. Roles: FEATURE = safe model input, LABEL = target, LEAK = only known after the outcome,
ID = identifier, PROVENANCE = describes where the row came from.

| Column | Role | Type | Missing | Meaning | Audit note |
|---|---|---|---|---|---|
| `row_id` | ID | id | 0.0% | Running row number in the sheet. | Meaningless for modelling; never a feature. |
| `employee_id` | ID | id | 0.0% | Employee code. EMP-xxxx = master-snapshot employee, HIST-xxxx = simulated history. | Use as the GROUP key when splitting data so one person is never in train and test. |
| `observation_date` | PROVENANCE | date | 0.0% | The day the features were measured (30 Sep 2025 for master-snapshot staff, 2021-2025 for history). | Use for time-aware splits, not as a feature. |
| `department_id` | FEATURE | category | 0.0% | Department code D00-D10 (D01 Engineering ... D10 Legal). | D00 Executive never appears in this table. |
| `role_id` | FEATURE | category | 0.0% | Job role code (48 distinct roles). | High-cardinality: encode carefully. |
| `job_level` | FEATURE | count | 0.0% | Seniority level 1 (associate) to 6 (head). | Level 6 has only 2 rows. |
| `work_location` | FEATURE | category | 0.0% | Office city. | 13 rows use messy spellings (BENGALURU, Madras ...). |
| `work_mode` | FEATURE | category | 0.0% | Onsite, Hybrid or Remote. |  |
| `employment_type` | FEATURE | category | 0.0% | Permanent or fixed-term contract. | Contract staff are only 6% of rows. |
| `highest_education` | FEATURE | category | 0.6% | Highest qualification, 27 free-text values. | 15 blanks. Many near-duplicate labels (B.E. vs B.Tech). |
| `tenure_months` | FEATURE | count | 0.0% | Months with the company at the observation date. | 4 impossible negative values. |
| `total_experience_years` | FEATURE | number | 0.0% | Years of work in total: earlier jobs plus time here. | Some values are negative or smaller than tenure. |
| `previous_employers_count` | FEATURE | count | 0.0% | How many employers the person had before joining. |  |
| `commute_distance_km` | FEATURE | number | 2.3% | Home-to-office distance in km. | 55 blanks. |
| `annual_fixed_pay` | FEATURE | number | 0.0% | Yearly fixed salary in INR. | Right-skewed; 3 values are 10x too large (typos). |
| `comp_ratio` | FEATURE | number | 0.0% | Pay divided by the role's mid-point; below 1 = underpaid. | Useful cross-check for pay typos. |
| `raises_received` | FEATURE | count | 0.0% | Number of pay revisions since joining. |  |
| `last_raise_pct` | FEATURE | number | 8.5% | Size of the most recent raise (%). | Blank = never had a raise (structural). |
| `months_since_last_raise` | FEATURE | count | 0.0% | Months since the last raise (or since joining if none). |  |
| `promotions_received` | FEATURE | count | 0.0% | Number of promotions since joining. |  |
| `months_since_last_promotion` | FEATURE | number | 61.1% | Months since the last promotion. | Blank = never promoted (structural, 61%). |
| `internal_transfers` | FEATURE | count | 0.0% | Moves between departments. |  |
| `last_performance_rating` | FEATURE | count | 7.6% | Latest appraisal rating, 1-5. | Blank = joined under 12 months ago, no appraisal yet. |
| `last_engagement_score` | FEATURE | number | 22.1% | Latest engagement survey score, 1-5. | Blank = did not respond (22%). |
| `manager_changes_12m` | FEATURE | count | 0.0% | Times the manager changed in the last year. |  |
| `training_hours_12m` | FEATURE | number | 0.0% | Training hours in the last year. | Long right tail, plausible. |
| `attendance_rate_3m` | FEATURE | share | 0.0% | Share of scheduled days attended, last 3 months (0-1). | 2 values above 1. |
| `late_arrivals_3m` | FEATURE | count | 0.0% | Late check-ins in the last 3 months. |  |
| `avg_overtime_hours_per_month_3m` | FEATURE | number | 0.0% | Average monthly overtime hours. | 81% are zero; max 50. |
| `avg_hours_worked_3m` | FEATURE | number | 0.0% | Average hours per working day. | One implausible low value (1.8h). |
| `wfh_share_3m` | FEATURE | share | 0.0% | Share of days worked from home (0-1). | Many values slightly below 0 or above 1. |
| `unpaid_leave_days_6m` | FEATURE | number | 0.0% | Unpaid leave days, last 6 months. |  |
| `paid_leave_days_6m` | FEATURE | count | 0.0% | Paid leave days, last 6 months. |  |
| `dept_attrition_rate_prior_12m` | FEATURE | share | 0.0% | Share of the department that left in the 12 months BEFORE the snapshot. | Legitimate (looks backwards), but confirm with the Python team. |
| `left_within_12m` | LABEL | label | 0.0% | TARGET: 1 if the person left within 12 months after the observation date. | 14.6% are 1. |
| `voluntary_left_within_12m` | LEAK | label | 0.0% | 1 if they resigned (rather than being let go). A sub-type of the target. | Known only once someone has left. |
| `source` | PROVENANCE | category | 0.0% | Where the row came from: master snapshot or historical extension. | Describes data origin, not the employee. Drop from features. |
| `exit_date` | LEAK | date | 85.4% | Last working day. Blank for people who stayed. | Filled for exactly the rows where target = 1. |
| `resignation_date` | LEAK | date | 86.6% | Day the resignation was submitted. | Only exists for people who resigned. |
| `exit_reason` | LEAK | category | 85.4% | Reason recorded when leaving. | Only exists for leavers. |
| `exit_interview_done` | LEAK | category | 85.4% | Y/N: was an exit interview held. | Only exists for leavers. |
| `fnf_settlement_amount` | LEAK | number | 85.4% | Full-and-final settlement paid at exit. | Only exists for leavers. |
