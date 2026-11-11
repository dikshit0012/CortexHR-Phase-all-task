"""Task 7 - plain-language `reason_text` for payroll clerks and HR.

Every sentence says (1) what looks unusual, (2) why it matters (the amount, when one can be estimated) and
(3) what to check next. No statistics vocabulary: nothing about models, scores, thresholds or features.
A statistical-model-only flag says the record is unusual and the cause is unknown; it never claims an error.
"""
from __future__ import annotations

import math
from typing import Callable, Dict

import pandas as pd

# Words that must never appear in a reason_text (checked by tests).
BANNED_TERMS = ("isolation", "forest", "model", "score", "threshold", "feature", "contamination",
                "algorithm", "statistic", "outlier", "predict", "z-score", "vector")


def _n(x) -> str:
    """12,345.67 style number, or 'an unknown amount' when the value is missing."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
            return "an unknown amount"
    except (TypeError, ValueError):
        return "an unknown amount"
    return f"{float(x):,.2f}"


def _g(x) -> str:
    """Short number without trailing zeros (days, hours)."""
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return "an unknown number of"


def _about(amount) -> str:
    return "" if amount is None or pd.isna(amount) else f" About {_n(amount)} may be affected."


def _month(period) -> str:
    return pd.Timestamp(period).strftime("%B %Y")


# Each function receives a dict of the record's values (see explain.py) and returns one sentence group.
def _duplicate(r: Dict) -> str:
    return (f"This employee has {int(r['copies'])} identical payslips for {_month(r['period'])}. "
            f"If one was paid in error, about {_n(r['dup_total'])} may have been paid twice. "
            f"That amount is shared equally across the matching payslips so it is not counted twice, so this one shows {_n(r['money'])}. "
            "Please check the payment records and cancel any duplicate.")


def _negative_net(r: Dict) -> str:
    return (f"The net pay on this payslip is negative ({_n(r['net_pay'])}): deductions of {_n(r['total_deductions'])} "
            f"are larger than the gross pay of {_n(r['gross_pay'])}."
            f"{_about(r['money'])} Please check the salary and deduction amounts before approving this payslip.")


def _zero_pay(r: Dict) -> str:
    return (f"The net pay is zero although the employee was due pay for {_g(r['payable_days'])} day(s).{_about(r['money'])} "
            "Please verify the salary amount and the deductions, and check whether unpaid leave was approved.")


def _gross_above(r: Dict) -> str:
    return (f"The gross pay ({_n(r['gross_pay'])}) is higher than the pay components on this payslip add up to "
            f"({_n(r['expected_gross'])}).{_about(r['money'])} Please check the fixed pay, overtime and bonus amounts.")


def _gross_below(r: Dict) -> str:
    return (f"The gross pay ({_n(r['gross_pay'])}) is lower than the pay components on this payslip add up to "
            f"({_n(r['expected_gross'])}).{_about(r['money'])} Please check the fixed pay, the days paid and whether unpaid days were approved.")


def _overtime_cap(r: Dict) -> str:
    return (f"{_g(r['overtime_hours_paid'])} overtime hours were paid, which is above the limit of {_g(r['overtime_cap'])} hours."
            f"{_about(r['money'])} Please check the overtime hours and whether an exception was approved.")


def _overtime_not_eligible(r: Dict) -> str:
    return (f"Overtime pay of {_n(r['overtime_pay'])} was paid to a role that is not eligible for overtime. "
            "Please verify the employee's role and the overtime approval.")


def _missed_pf(r: Dict) -> str:
    return (f"No provident fund was deducted this month, but {_n(r['prev_pf_employee'])} was deducted last month.{_about(r['money'])} "
            "Please check whether the employee's provident fund enrolment changed.")


def _role_norm(r: Dict) -> str:
    return (f"The fixed monthly pay ({_n(r['monthly_fixed_gross'])}) is more than twice the usual pay for this role "
            f"({_n(r['role_median_fixed_gross'])}).{_about(r['money'])} Please verify the employee's grade and salary record.")


def _salary_jump(r: Dict) -> str:
    return (f"The fixed monthly pay rose by more than half compared with last month ({_n(r['prev_monthly_fixed_gross'])} to "
            f"{_n(r['monthly_fixed_gross'])}).{_about(r['money'])} Please check that an approved salary revision exists.")


def _salary_drop(r: Dict) -> str:
    context = (" Last month's pay was itself flagged as unusually high, so this may simply be a return to the normal level."
               if r.get("prev_month_spike") else "")
    return (f"The fixed monthly pay fell compared with last month ({_n(r['prev_monthly_fixed_gross'])} to "
            f"{_n(r['monthly_fixed_gross'])}).{_about(r['money'])}{context} "
            "Please check that this was an approved change and that the salary record is up to date.")


def _invalid(r: Dict) -> str:
    return ("One or more payroll amounts on this payslip are missing or are not valid numbers, so it cannot be trusted. "
            "Please correct the payslip data before it is approved.")


def _payroll_unusual(r: Dict) -> str:
    return ("This payslip looks different from the usual payroll pattern, although no specific payroll check failed, "
            "and the reason is not known. No amount at risk has been estimated. Please review the payslip.")


def _no_attendance(r: Dict) -> str:
    return (f"No attendance was recorded for the whole month although {_g(r['scheduled_days'])} working days were scheduled "
            "and paid leave does not cover them. Please check whether the employee was absent, on leave that was not recorded, "
            "or missing from the attendance system.")


def _long_unpaid(r: Dict) -> str:
    return (f"The employee had {_g(r['longest_unpaid_streak_days'])} unpaid absence days in a row this month. "
            "Please check whether the leave was approved and recorded correctly.")


def _hours(r: Dict) -> str:
    return (f"The average working day this month was {float(r['avg_hours_worked']):.1f} hours, which is more than 12 hours a day. "
            "Please check the clock-in and clock-out records for entry mistakes.")


def _pattern(r: Dict) -> str:
    return ("This month's attendance looks different from other employees and from this person's usual pattern, "
            "and the reason is not known. Please review the attendance record.")


TEXT_BUILDERS: Dict[str, Callable[[Dict], str]] = {
    "DUPLICATE_PAYMENT": _duplicate, "NEGATIVE_NET_PAY": _negative_net, "ZERO_PAY": _zero_pay,
    "GROSS_ABOVE_NORM": _gross_above, "GROSS_BELOW_EXPECTED": _gross_below, "OVERTIME_ABOVE_CAP": _overtime_cap,
    "OVERTIME_NOT_ELIGIBLE": _overtime_not_eligible, "PF_NOT_DEDUCTED": _missed_pf, "PAY_ABOVE_ROLE_NORM": _role_norm,
    "SALARY_JUMP": _salary_jump, "SALARY_DROP": _salary_drop, "INVALID_PAYROLL_DATA": _invalid,
    "STATISTICAL_OUTLIER": _payroll_unusual,
    "NO_ATTENDANCE": _no_attendance, "LONG_UNPAID_ABSENCE": _long_unpaid, "IMPLAUSIBLE_HOURS": _hours,
    "PATTERN_SHIFT": _pattern,
}


def build_reason_text(code: str, record: Dict) -> str:
    """Reason sentence for one record; unknown codes raise (never silently skipped)."""
    if code not in TEXT_BUILDERS:
        raise KeyError(f"no reason text defined for reason code {code!r}")
    return TEXT_BUILDERS[code](record)
