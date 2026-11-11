# Task 7 - Explain Every Flag: Reason Codes and Money at Risk (CortexHR, Phase 2, Day 7)

> **Fictional data.** All people, payslips and amounts are fictional (Zenith Nova Technologies, part of Aura Workforce).
> **Human review is PENDING.** The 20 explanations have not been read by a real person yet (see "Human-review status").
> **Evaluation labels:** `unlock_CX-06` is missing from the workbook. Task 7 outputs never use labels; the two optional
> evaluation-only checks use the substitute `payroll_anomaly_labels`, so they are provisional.
> **Workspace limitation:** the repository includes the Task 5 and Task 6 source, but not the source workbook.
> Task 7's synthetic tests can run without it; the end-to-end runner and workbook-backed integration tests need
> an authorized local workbook. The existing reports are retained artifacts; their real-data figures have not
> been independently revalidated here.

## Objective
Task 5 (attendance) and Task 6 (payroll) say *which* records look unusual. Task 7 tells a payroll clerk **why** and **how much money may be affected**:
every flag gets a stable `reason_code`, a plain-language `reason_text`, and (for payroll flags where an amount can be supported) an
estimated `money_at_risk`. Flags are ranked by that amount and written in the official format of the workbook sheet `sample_anomaly_results`.
Task 7 **explains the existing flags**. It runs the Task 5 and Task 6 detectors unchanged and does not retrain, tune or modify any model.

## Project structure
```
task 7/
  src/cortexhr_task7/   reason_codes.py (dictionary + priority)   reason_text.py (clerk-friendly sentences)
                        money_at_risk.py (formulas)   explain.py (signals -> reasons)   ranking.py   official_output.py (schema)
                        pipeline.py (end to end + integrity checks)   review_pack.py (20-row sample, review sheet)
                        evaluation_check.py (EVALUATION-ONLY, the only module that reads labels)
  scripts/              run_task7.py   verify_task7_reports.py (independent recomputation)
  tests/                synthetic tests (no real payroll or attendance records)
  reports/              official output, 20-row sample, detail report, dictionary, review sheet, summary, evaluation-only tables
  docs/                 HUMAN_REVIEW_CHECKLIST.md   REVIEWER_FEEDBACK_TEMPLATE.md
```

## Install
```bash
python -m pip install -r "phase-2/task07/task 7/requirements.txt"     # pandas, numpy, scikit-learn, openpyxl, pytest
```
The workbook `02_CortexHR_AI_ML_Dataset.xlsx` is found automatically (same search order as Task 6: `--workbook`, `CORTEXHR_WORKBOOK`, `phase-2/task06/task 6/data/raw`, `phase-2/task05/data/raw`, ...).

## Run (from the repository root)
```bash
python -m pytest "phase-2/task05/tests"                              # regression
python -m pytest "phase-2/task06/task 6/tests"                       # regression
python -m pytest "phase-2/task07/task 7/tests"                       # Task 7 tests
python "phase-2/task07/task 7/scripts/run_task7.py"                  # builds every file in phase-2/task07/task 7/reports/
python "phase-2/task07/task 7/scripts/verify_task7_reports.py"       # ids, amounts, full ranking, exact 20-row sample
```
Options: `--scored-at "YYYY-MM-DD HH:MM:SS"` (fixed by default so reruns are identical), `--sample-size 20`, `--skip-label-check`.
The verifier accepts the matching expected size with `--sample-size`; pass the same value used by the runner when it is not 20.

## Official output schema
Source of truth: the header of the workbook sheet `sample_anomaly_results`. The script re-reads that header on every run and stops if it differs.
Columns, in this exact order:

`result_id` , `source_dataset` , `record_id` , `employee_id` , `period` , `anomaly_score` , `flagged` , `reason_code` , `reason_text` , `money_at_risk` , `model_version` , `scored_at`

| Column | How Task 7 fills it |
|---|---|
| `result_id` | `AN-0001`, `AN-0002`, ... in ranking order |
| `source_dataset` | `payroll_monthly_features` (Task 6 flags) or `attendance_monthly_features` (Task 5 flags) |
| `record_id`, `employee_id`, `period` | Taken from the flagged source record (`YYYY-MM-DD`). These are the fictional ids the schema requires |
| `anomaly_score` | The detector's Isolation Forest score scaled to 0-1 over all rows of that dataset (min-max, 3 decimals). **Rule-flagged rows can have a low score** because the score comes from the model only |
| `flagged` | `Y`. The file contains only flagged records. A record is flagged by the Task 5/6 combined detector (rules **or** model) |
| `reason_code`, `reason_text` | The **primary** reason and its plain-language sentence |
| `money_at_risk` | Estimated exposure in INR, `>= 0`; **empty** when no amount can be supported |
| `model_version` | `iforest+rules-payroll-task6` or `iforest+rules-attendance-task5` (descriptive labels chosen here) |
| `scored_at` | Run parameter, default `2026-10-01 02:00:00` (the sample's batch time) so reruns are byte-identical |

Internal columns (all reasons, signal counts, candidate amounts, masked employee reference) are **not** in the official file; they are in `reports/task7_flag_details.csv`.

## Reason-code dictionary
Codes marked "yes" already exist in the official sample's vocabulary (same spelling). Codes marked **new** do not; the Data team needs to confirm they are accepted.
`priority` is the order used to choose the **primary** reason when several signals fire, and to break ranking ties. It is a documented business ordering, not a probability.

| priority | code | source | detector signal | meaning | in official sample |
|---|---|---|---|---|---|
| 1 | `DUPLICATE_PAYMENT` | payroll | rule_duplicate_payment | Same employee, same pay month and identical amounts appear on more than one payslip. | **new** |
| 2 | `NEGATIVE_NET_PAY` | payroll | rule_negative_net | Net pay is below zero. | yes |
| 3 | `ZERO_PAY` | payroll | rule_zero_net | Net pay is exactly zero although the employee had payable days. | yes |
| 4 | `GROSS_ABOVE_NORM` | payroll | rule_gross_mismatch (gross higher) | Gross pay is higher than fixed pay for the days paid + overtime + variable pay add up to. | yes |
| 5 | `GROSS_BELOW_EXPECTED` | payroll | rule_gross_mismatch (gross lower) | Gross pay is lower than fixed pay for the days paid + overtime + variable pay add up to. | yes |
| 6 | `OVERTIME_ABOVE_CAP` | payroll | rule_overtime_cap (hours above cap) | Paid overtime hours exceed the 50-hour policy cap. | yes |
| 7 | `OVERTIME_NOT_ELIGIBLE` | payroll | rule_overtime_cap (role not eligible) | Overtime was paid to a role that is not overtime-eligible. | **new** |
| 8 | `PF_NOT_DEDUCTED` | payroll | rule_missed_pf | Provident fund is 0 although it was deducted in the previous month. | yes |
| 9 | `PAY_ABOVE_ROLE_NORM` | payroll | rule_role_pay_outlier | Fixed pay is more than twice the median fixed pay of the same role. | **new** |
| 10 | `SALARY_JUMP` | payroll | rule_fixed_gross_jump | Fixed pay rose by more than 50% compared with the previous month. | **new** |
| 11 | `SALARY_DROP` | payroll | rule_salary_drop | Fixed pay fell compared with the previous month. | **new** |
| 12 | `INVALID_PAYROLL_DATA` | payroll | rule_invalid_amounts | A required payroll amount is missing or not a valid number. | **new** |
| 13 | `NO_ATTENDANCE` | attendance | rule_unexcused_absence (no presence) | No presence recorded in a month with scheduled days and not covered by paid leave. | yes |
| 14 | `LONG_UNPAID_ABSENCE` | attendance | rule_unexcused_absence (long streak) | Five or more consecutive unpaid absence days. | yes |
| 15 | `IMPLAUSIBLE_HOURS` | attendance | rule_implausible_hours | Average working day above 12 hours. | **new** |
| 16 | `STATISTICAL_OUTLIER` | payroll | Isolation Forest only (payroll) | Flagged only by the statistical model: the payslip is unusual, no rule was broken, the cause is unknown. | yes |
| 17 | `PATTERN_SHIFT` | attendance | Isolation Forest only (attendance) | Flagged only by the statistical model: attendance is unusual, no rule was broken, the cause is unknown. | yes |

Every Task 5 and Task 6 rule and both model signals are covered; a test fails if a rule is added without a mapping, and the script stops (not silently skips) on an unmapped rule or a flag it cannot explain.
Model-only flags say only that the record looks different from the usual pattern and that the reason is not known.

### Flags in the generated output
| reason code | rows |
|---|--:|
| `DUPLICATE_PAYMENT` | 30 |
| `NEGATIVE_NET_PAY` | 8 |
| `ZERO_PAY` | 5 |
| `GROSS_ABOVE_NORM` | 18 |
| `GROSS_BELOW_EXPECTED` | 12 |
| `OVERTIME_ABOVE_CAP` | 8 |
| `PF_NOT_DEDUCTED` | 14 |
| `PAY_ABOVE_ROLE_NORM` | 12 |
| `SALARY_DROP` | 13 |
| `NO_ATTENDANCE` | 15 |
| `LONG_UNPAID_ABSENCE` | 48 |
| `IMPLAUSIBLE_HOURS` | 35 |
| `STATISTICAL_OUTLIER` | 133 |
| `PATTERN_SHIFT` | 156 |

## Money at risk
**Meaning:** an *estimate of the money that may be affected if the flagged record is wrong* (overpayment, underpayment or over-deduction).
It is exposure to verify, **not confirmed loss**; an unusual payslip is not proof that the company lost this amount. Do not add the column up and call it a loss.

**Rules for every formula:** never negative; rounded to 2 decimals; missing, zero-denominator or non-finite inputs give an **empty** amount (never 0 for "unknown");
only the **primary** reason's amount is shown, so overlapping signals are never added together (other candidate amounts are kept in the detail report).

| reason code | amount |
|---|---|
| `DUPLICATE_PAYMENT` | Possible extra payment = (copies - 1) x net pay, split equally over the flagged copies so it is never counted twice. |
| `NEGATIVE_NET_PAY` | Deductions above the employee's usual deduction share of gross pay (own previous months; dataset median if none). |
| `ZERO_PAY` | Expected net pay = expected gross pay x (1 - usual deduction share); amount not paid. |
| `GROSS_ABOVE_NORM` | gross pay - expected gross pay. |
| `GROSS_BELOW_EXPECTED` | expected gross pay - gross pay. |
| `OVERTIME_ABOVE_CAP` | Overtime pay x (hours - cap) / hours: pay attributable to the hours above the cap. |
| `OVERTIME_NOT_ELIGIBLE` | Overtime pay paid. |
| `PF_NOT_DEDUCTED` | Provident fund deducted in the previous month. |
| `PAY_ABOVE_ROLE_NORM` | (fixed pay - role median fixed pay) x payable days / days in month. |
| `SALARY_JUMP` | (fixed pay - last month's fixed pay) x payable days / days in month. |
| `SALARY_DROP` | (last month's fixed pay - fixed pay) x payable days / days in month. |
| `INVALID_PAYROLL_DATA` | No estimate (the amounts needed are not reliable). |
| `NO_ATTENDANCE` | Not applicable (attendance rows carry no money estimate). |
| `LONG_UNPAID_ABSENCE` | Not applicable (attendance rows carry no money estimate). |
| `IMPLAUSIBLE_HOURS` | Not applicable (attendance rows carry no money estimate). |
| `STATISTICAL_OUTLIER` | No estimate. Left empty because no defensible amount exists. |
| `PATTERN_SHIFT` | Not applicable (attendance rows carry no money estimate). |

**Assumptions.** *Usual deduction share* = the employee's median deductions/gross over **previous** calendar months, else the dataset-wide median (label-free, uses only the past).
*Expected gross* = fixed pay x payable days / days in month + overtime pay + variable pay (the same identity Task 6 checks).
*Prorating* = payable days / days in month. For negative net pay the amount is never less than |net pay| (the minimum by which deductions must be wrong).
A duplicate's total (copies - 1) x net pay is shared equally across the flagged copies, so the shares add up to one extra payment and the data does not need to say which copy is the extra one.
**Empty amounts:** statistical-model-only flags, invalid-data flags and all attendance flags. The sample's own payroll rows always have a number; leaving these empty is a deliberate choice (see Limitations).

## Ranking
Sort keys in order: (1) `money_at_risk`, largest first, empty amounts last; (2) reason priority (more serious type of issue first); (3) number of independent signals (rules + model);
(4) the detector's raw score; (5) `record_id` ascending. Because the last key is unique, the order never depends on the order of the input rows (tested by shuffling).
**Amount, severity and confidence are different:** the amount says how much money could be affected; the priority says how serious the *type* of issue is; the signal count says how much evidence there is.
None of them is a probability, and a large amount does not make an alert more likely to be a real error.

## Example output (official columns, `employee_id` omitted here)
| result_id | record_id | period | reason_code | money_at_risk | reason_text |
|---|---|---|---|--:|---|
| AN-0010 | PI-005869 | 2026-06-01 | `NEGATIVE_NET_PAY` | 621,266.24 | The net pay on this payslip is negative (-174,244.25): deductions of 688,444.91 are larger than the gross pay of 514,200.66. About 621,266.24 may be affected. Please check the salary and deduction amounts before approving this payslip. |
| AN-0042 | PI-002412 | 2025-10-01 | `DUPLICATE_PAYMENT` | 136,061.73 | This employee has 2 identical payslips for October 2025. If one was paid in error, about 272,123.47 may have been paid twice. That amount is shared equally across the matching payslips so it is not counted twice, so this one shows 136,061.73. Please check the payment records and cancel any duplicate. |
| AN-0136 | AF-03098 | 2025-07-01 | `LONG_UNPAID_ABSENCE` |  | The employee had 13 unpaid absence days in a row this month. Please check whether the leave was approved and recorded correctly. |
| AN-0219 | PI-004863 | 2026-04-01 | `STATISTICAL_OUTLIER` |  | This payslip looks different from the usual payroll pattern, although no specific payroll check failed, and the reason is not known. No amount at risk has been estimated. Please review the payslip. |

The 20-row evidence file is `reports/task7_sample_20_flags.csv` (16 payroll and 4 attendance flags). It is chosen by a fixed rule: the best-ranked flag of every reason code present
(14 codes), then the next best-ranked payroll flags with an amount. All rows are real flags from the run; none was invented or repeated.

## Previously reported test results
The following results came from a prior run and are **not verified in this workspace**. Task 5 and Task 6 source are now
included in this checkout, but the workbook needed for real-data runs is absent. The prior report described a Linux run;
this checkout is on Windows.

| Command | Result |
|---|---|
| `python -m pytest "phase-2/task05/tests"` | Source is included; run from the repository root |
| `python -m pytest "phase-2/task06/task 6/tests"` | Source is included; run from the repository root |
| `python -m pytest "phase-2/task07/task 7/tests"` | Source is included; run from the repository root |
| `python "phase-2/task07/task 7/scripts/run_task7.py"` | Requires the Task 5/6 source and an authorized local workbook |
| `python "phase-2/task07/task 7/scripts/verify_task7_reports.py"` | Real-report verification requires the workbook |

The verifier also checks deterministic tie-break ordering and requires the sample to match the reason-coverage selection rule.
Standalone synthetic verifier tests are in `tests/test_verifier_integrity.py`; they avoid Task 5 and Task 6.
In this audit, those six standalone tests passed under pytest with `--noconftest`. The ordinary suite still cannot
collect because its shared `conftest.py` imports the absent Task 6 package.

Task 7 tests cover: exact schema (names, order, formats, and the real sample sheet when available); reason mapping for every rule and model signal;
every formula against a hand-computed number; duplicates not double-counted; negative/zero net pay; missing, zero-denominator and invalid inputs;
non-negative amounts; ranking and tie-breaks; row-shuffle stability; no flag dropped or invented; wrong employee/period detected; no label use;
the end-to-end script and its reproducibility. Deliberately breaking the code (six ways) made the tests fail each time.

## Evaluation-only checks (do not influence any output)
`evaluation_check.py` runs **after** the outputs are written, on the substitute labels, and answers whether the reasons and amounts are sensible
(`reports/task7_evalonly_reason_vs_label.csv`, `task7_evalonly_money_vs_reference.csv`). Findings:
- Reason codes agree with the injected error types (for example 8/8 negative-net, 12/12 stale-salary flagged as `GROSS_BELOW_EXPECTED`, all 14 missed-PF flags carry the missed-deduction label).
- Amounts match a reference rebuilt from the label's original value within 10% for 15/15 double payments, 15/15 wrong-decimal, 8/8 negative-net, 12/12 stale-salary, 13/14 missed-PF.
- Weaker: role mismatch 11/15 within 10% (the estimate uses the role median, not the original salary); **overtime spikes 0/8**: the amount counts only pay for hours **above the 50-hour cap**, so it is about 0.42 of the injected change by design.
- All 13 `SALARY_DROP` flags and the 15 legitimate originals of duplicate pairs are false alarms by these labels; the 13 salary-drop alerts carry large amounts and rank high (see Limitations).
These agreements are partly expected: the Task 6 rules were designed knowing the error types, and the injected errors are crisp. They are not proof of performance on real payroll.

## Known limitations
1. **`unlock_CX-06` is missing.** Task 7 itself needs no labels. The evaluation-only checks use `payroll_anomaly_labels` and are provisional.
2. **Seven reason codes are new** (`DUPLICATE_PAYMENT`, `OVERTIME_NOT_ELIGIBLE`, `PAY_ABOVE_ROLE_NORM`, `SALARY_JUMP`, `SALARY_DROP`, `INVALID_PAYROLL_DATA`, `IMPLAUSIBLE_HOURS`). Confirm with the Data team that codes outside the sample vocabulary are accepted.
   Also, `GROSS_ABOVE_NORM` here means "gross is above what the payslip's own components add up to"; the sample describes it as a multiple of fixed pay.
3. **Empty amounts.** 387 of 507 rows have no `money_at_risk` (133 model-only payroll flags, 254 attendance flags). The sample's payroll rows always carry a number; confirm that an empty value is acceptable for payroll rows with no supportable estimate.
4. **Estimates, not losses.** The formulas rest on the assumptions above. The overtime amount deliberately counts only the above-cap share and so understates.
5. **Duplicates.** No payment or bank reference exists, so these are *possible* duplicates; both identical payslips are flagged (Task 6 default) and the amount is shared between them.
6. **`SALARY_DROP`.** The Task 6 rule had no true positives on the substitute labels. Its alerts can carry large amounts and rank high even when the drop is just a return to normal after an earlier spike. The text says so when the previous month was itself flagged, but the ranking is unchanged. Task 6 was not modified.
7. **`anomaly_score` and `flagged`.** The score comes from the model only, so rule-flagged rows can score low; `flagged = Y` means "flagged by rules or model", not "above a model threshold".
8. **Task 5 row-order dependence.** The Task 5 model fit depends on input row order (a shuffle of the attendance file changed 26 model flags and 0 rule flags in a check). Task 5 was not modified, so attendance `PATTERN_SHIFT` flags can change if the file is reordered. Task 6 is order independent.
9. **Identifiers.** The schema requires `employee_id` and `record_id`; the official files contain these fictional ids (already present in the public workbook). The detail report and review sheet use masked employee references or none.
10. `scored_at` and `model_version` are run parameters / descriptive labels chosen for this task.

## Human-review status
**PENDING.** The 20 explanations have not been reviewed by a real person, and no feedback is recorded anywhere in this project.
`reports/task7_human_review_sheet.csv` has blank human columns. Give it with `docs/HUMAN_REVIEW_CHECKLIST.md` to a reader who knows payroll but did not build this, record their answers in `docs/REVIEWER_FEEDBACK_TEMPLATE.md`,
and only then change this status. The `automated_lint` column (all 20 PASS) checks for technical words, a next step and length; it is not a substitute.
