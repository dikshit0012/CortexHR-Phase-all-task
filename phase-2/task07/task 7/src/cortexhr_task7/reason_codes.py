"""Task 7 - the reason-code dictionary: one stable code per detector signal.

Vocabulary: codes that already exist in the official `sample_anomaly_results` sheet are reused with the
same spelling (NEGATIVE_NET_PAY, ZERO_PAY, GROSS_ABOVE_NORM, GROSS_BELOW_EXPECTED, OVERTIME_ABOVE_CAP,
PF_NOT_DEDUCTED, NO_ATTENDANCE, LONG_UNPAID_ABSENCE, STATISTICAL_OUTLIER, PATTERN_SHIFT). Codes the sample
does not contain are new and marked `in_official_sample = False`.

PRIORITY is the deterministic order used to pick the PRIMARY reason when several signals fire on one record
and to break ranking ties. It is a documented business ordering, NOT a probability or a confidence value:
  1. hard impossibilities (duplicate payment, negative or zero net pay)
  2. payslip does not add up (gross vs its own components)
  3. policy limits (overtime cap, provident fund)
  4. comparison with peers / history (role norm, salary jump, salary drop)
  5. bad input data
  6. attendance rules, then
  7. the statistical model, which only says "unusual", never "wrong".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class ReasonCode:
    code: str
    source: str               # "payroll" or "attendance"
    signal: str               # the detector signal that produces it
    meaning: str              # stable meaning of the code
    money_basis: str          # how money_at_risk is estimated for it (see money_at_risk.py)
    in_official_sample: bool


_CODES: List[ReasonCode] = [
    ReasonCode("DUPLICATE_PAYMENT", "payroll", "rule_duplicate_payment",
               "Same employee, same pay month and identical amounts appear on more than one payslip.",
               "Possible extra payment = (copies - 1) x net pay, split equally over the flagged copies so it is never counted twice.", False),
    ReasonCode("NEGATIVE_NET_PAY", "payroll", "rule_negative_net",
               "Net pay is below zero.",
               "Deductions above the employee's usual deduction share of gross pay (own previous months; dataset median if none).", True),
    ReasonCode("ZERO_PAY", "payroll", "rule_zero_net",
               "Net pay is exactly zero although the employee had payable days.",
               "Expected net pay = expected gross pay x (1 - usual deduction share); amount not paid.", True),
    ReasonCode("GROSS_ABOVE_NORM", "payroll", "rule_gross_mismatch (gross higher)",
               "Gross pay is higher than fixed pay for the days paid + overtime + variable pay add up to.",
               "gross pay - expected gross pay.", True),
    ReasonCode("GROSS_BELOW_EXPECTED", "payroll", "rule_gross_mismatch (gross lower)",
               "Gross pay is lower than fixed pay for the days paid + overtime + variable pay add up to.",
               "expected gross pay - gross pay.", True),
    ReasonCode("OVERTIME_ABOVE_CAP", "payroll", "rule_overtime_cap (hours above cap)",
               "Paid overtime hours exceed the 50-hour policy cap.",
               "Overtime pay x (hours - cap) / hours: pay attributable to the hours above the cap.", True),
    ReasonCode("OVERTIME_NOT_ELIGIBLE", "payroll", "rule_overtime_cap (role not eligible)",
               "Overtime was paid to a role that is not overtime-eligible.",
               "Overtime pay paid.", False),
    ReasonCode("PF_NOT_DEDUCTED", "payroll", "rule_missed_pf",
               "Provident fund is 0 although it was deducted in the previous month.",
               "Provident fund deducted in the previous month.", True),
    ReasonCode("PAY_ABOVE_ROLE_NORM", "payroll", "rule_role_pay_outlier",
               "Fixed pay is more than twice the median fixed pay of the same role.",
               "(fixed pay - role median fixed pay) x payable days / days in month.", False),
    ReasonCode("SALARY_JUMP", "payroll", "rule_fixed_gross_jump",
               "Fixed pay rose by more than 50% compared with the previous month.",
               "(fixed pay - last month's fixed pay) x payable days / days in month.", False),
    ReasonCode("SALARY_DROP", "payroll", "rule_salary_drop",
               "Fixed pay fell compared with the previous month.",
               "(last month's fixed pay - fixed pay) x payable days / days in month.", False),
    ReasonCode("INVALID_PAYROLL_DATA", "payroll", "rule_invalid_amounts",
               "A required payroll amount is missing or not a valid number.",
               "No estimate (the amounts needed are not reliable).", False),
    ReasonCode("NO_ATTENDANCE", "attendance", "rule_unexcused_absence (no presence)",
               "No presence recorded in a month with scheduled days and not covered by paid leave.",
               "Not applicable (attendance rows carry no money estimate).", True),
    ReasonCode("LONG_UNPAID_ABSENCE", "attendance", "rule_unexcused_absence (long streak)",
               "Five or more consecutive unpaid absence days.",
               "Not applicable (attendance rows carry no money estimate).", True),
    ReasonCode("IMPLAUSIBLE_HOURS", "attendance", "rule_implausible_hours",
               "Average working day above 12 hours.",
               "Not applicable (attendance rows carry no money estimate).", False),
    ReasonCode("STATISTICAL_OUTLIER", "payroll", "Isolation Forest only (payroll)",
               "Flagged only by the statistical model: the payslip is unusual, no rule was broken, the cause is unknown.",
               "No estimate. Left empty because no defensible amount exists.", True),
    ReasonCode("PATTERN_SHIFT", "attendance", "Isolation Forest only (attendance)",
               "Flagged only by the statistical model: attendance is unusual, no rule was broken, the cause is unknown.",
               "Not applicable (attendance rows carry no money estimate).", True),
]

REASON_CODES: Dict[str, ReasonCode] = {c.code: c for c in _CODES}
PRIORITY: List[str] = [c.code for c in _CODES]          # documented order = list order
PRIORITY_INDEX: Dict[str, int] = {code: i for i, code in enumerate(PRIORITY)}


def reason_code_dictionary() -> pd.DataFrame:
    """Reason-code documentation table (also saved as a CSV)."""
    return pd.DataFrame([{
        "priority": i + 1, "reason_code": c.code, "source": c.source, "detector_signal": c.signal,
        "meaning": c.meaning, "money_at_risk_basis": c.money_basis, "in_official_sample": c.in_official_sample,
    } for i, c in enumerate(_CODES)])
