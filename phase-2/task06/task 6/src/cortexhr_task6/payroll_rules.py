"""Task 6 - transparent payroll business rules (no model, no labels).

Every rule returns a 0/1 column. Rules are "candidate checks": a flag means
"a human should look", not "this is proven wrong".

Thresholds come from business constraints in the data dictionary / payroll policy
or from label-free reference data, and are NOT tuned on the evaluation labels.
Change them through `RuleConfig`, no source edit needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .payroll_features import EMPLOYEE_COL, ID_COL, MONTH_COL, MONEY_COLUMNS, month_period


@dataclass
class RuleConfig:
    # Data dictionary: "Overtime hours paid (capped at 50)".
    overtime_cap_hours: float = 50.0
    # Data dictionary: gross = earned fixed + overtime + variable. 0.1% allows for rounding;
    # 99.5% of rows reconcile to < 0.001% (label-free check).
    gross_tolerance: float = 0.001
    # A salary more than 2x the previous month's role median is a grade/role mismatch candidate
    # (label-free: 99.5% of rows are <= 1.70x the prior-month role median).
    role_ratio_limit: float = 2.0
    # Fixed salary should only change through a revision; a rise of > 50% in one month
    # (ratio > 1.5) is implausible without a promotion record (not in this dataset).
    fixed_gross_jump_limit: float = 1.5
    # Any fall of more than 1% in fixed gross vs the previous month: salaries are revised
    # upward, so a fall suggests payment on a stale/old salary.
    fixed_gross_drop_limit: float = 0.99
    # Which payslips of a suspected duplicate group are flagged (all choices ignore input row order):
    #   "all"              - every payslip in the group (DEFAULT): the data has no payment reference or
    #                        timestamp saying which copy is the extra one, so the whole group is
    #                        flagged for review.
    #   "later_by_item_id" - every copy except the one with the smallest payroll_item_id. Only valid if
    #                        item ids are issued in processing order; an id prefix can break that.
    #   "file_order"       - ORIGINAL EXPERIMENT ONLY: later copy in input-file order. Depends on row order.
    duplicate_selection: str = "all"
    # POST-EVALUATION EXPERIMENT (default OFF, not adopted): do not flag a salary drop when the previous
    # month was flagged as a salary spike and this month returns to the pre-spike level.
    suppress_drop_after_flagged_spike: bool = False
    reversion_tolerance: float = 0.01      # "returns to the pre-spike level" = within 1% (mirrors the 1% drop limit)


DUPLICATE_SELECTIONS = ("all", "later_by_item_id", "file_order")


RULE_COLUMNS: List[str] = [
    "rule_duplicate_payment",
    "rule_negative_net",
    "rule_zero_net",
    "rule_gross_mismatch",
    "rule_overtime_cap",
    "rule_missed_pf",
    "rule_salary_drop",
    "rule_fixed_gross_jump",
    "rule_role_pay_outlier",
    "rule_invalid_amounts",
]

RULE_REASONS: Dict[str, str] = {
    "rule_duplicate_payment": "possible duplicate payment (same employee, month and identical amounts as another payslip)",
    "rule_negative_net": "negative net pay",
    "rule_zero_net": "zero net pay although the employee had payable days",
    "rule_gross_mismatch": "gross pay does not reconcile with fixed pay + overtime + variable pay",
    "rule_overtime_cap": "overtime hours above the policy cap (or overtime paid to a non-eligible role)",
    "rule_missed_pf": "provident fund is 0 although it was deducted last month",
    "rule_salary_drop": "fixed gross fell versus the previous month (possible stale salary)",
    "rule_fixed_gross_jump": "fixed gross jumped by more than 50% versus the previous month",
    "rule_role_pay_outlier": "fixed gross is more than 2x the median for the role",
    "rule_invalid_amounts": "a required payroll amount is missing or not a finite number",
}


def rule_definitions(cfg: RuleConfig) -> pd.DataFrame:
    """Documentation table: rule name, condition, business reason, false positives, output flag."""
    rows = [
        ("rule_duplicate_payment",
         "Same employee_id + month AND identical gross_pay, total_deductions and net_pay as at least one other payslip. "
         f"Flagged: {cfg.duplicate_selection!r} (all = every payslip in the group; the data cannot say which copy is the extra one).",
         "A person should be paid once per month.",
         "Genuine off-cycle or corrected payslips with the same amounts; when all copies are flagged, the legitimate original is flagged too.",
         "Potential duplicate - needs confirmation against payment/bank references (none in this dataset)."),
        ("rule_negative_net", "net_pay < 0",
         "Take-home pay cannot be negative.",
         "Recovery of a prior overpayment could legitimately net negative.", "rule_negative_net"),
        ("rule_zero_net", "net_pay == 0 AND payable_days > 0",
         "An employee with payable days should receive pay; fully unpaid months have payable_days = 0.",
         "Unrecorded unpaid leave or full-and-final settlement months.", "rule_zero_net"),
        ("rule_gross_mismatch",
         f"|gross_pay - expected_gross| / expected_gross > {cfg.gross_tolerance}  (expected = fixed*payable_days/days_in_month + overtime + variable)",
         "Data-dictionary identity for gross pay; breaks when an amount is mis-keyed (e.g. a shifted decimal).",
         "Rows where gross includes components not in this dataset (arrears, adjustments).", "rule_gross_mismatch"),
        ("rule_overtime_cap", f"overtime_hours_paid > {cfg.overtime_cap_hours:g} OR (overtime_hours_paid > 0 AND overtime_eligible != 'Y')",
         "Policy caps overtime at 50 hours and only eligible roles earn it.",
         "Approved exceptions to the cap.", "rule_overtime_cap"),
        ("rule_missed_pf", "pf_employee == 0 AND gross_pay > 0 AND the employee's previous-month pf_employee > 0",
         "PF is a statutory monthly deduction; a stop without history of exemption is suspicious.",
         "PF opt-out after a wage change; first month cannot be checked (no history).", "rule_missed_pf"),
        ("rule_salary_drop", f"monthly_fixed_gross < {cfg.fixed_gross_drop_limit} x monthly_fixed_gross of the previous CALENDAR month (no flag if that month is missing)",
         "Salaries are revised upward; a drop suggests payment on the pre-revision salary.",
         "Real salary cuts/demotions; the month AFTER an erroneous spike also looks like a drop.", "rule_salary_drop"),
        ("rule_fixed_gross_jump", f"monthly_fixed_gross > {cfg.fixed_gross_jump_limit} x previous-month monthly_fixed_gross",
         "A >50% one-month rise needs a promotion record; none exists in this dataset.",
         "Promotions with large increments.", "rule_fixed_gross_jump"),
        ("rule_role_pay_outlier", f"monthly_fixed_gross > {cfg.role_ratio_limit} x median monthly_fixed_gross of the same role_id in the previous calendar month",
         "Pay should be close to the prior month's role norm; double is a role/grade mismatch candidate.",
         "Genuine outliers (rare skills, retention packages); roles with very few holders.", "rule_role_pay_outlier"),
        ("rule_invalid_amounts", "Any required money / day column is NaN or non-finite",
         "A payslip with a missing or invalid amount cannot be trusted.", "Intentionally blank optional fields (none among required columns).",
         "rule_invalid_amounts"),
    ]
    return pd.DataFrame(rows, columns=["Rule name", "Condition", "Business reason", "Potential false positives", "Output flag"])


def detect_duplicate_payments(df: pd.DataFrame, selection: str = "all") -> pd.Series:
    """Flag payslips that repeat another payslip: same employee + month + identical amounts.

    Equal salary alone is never enough - the employee and pay month must match too.
    The data has no payment reference, so a flag means "possible duplicate", not "proven duplicate".
    `selection` decides which members of a duplicate group are flagged (see RuleConfig).
    """
    if selection not in DUPLICATE_SELECTIONS:
        raise ValueError(f"duplicate_selection must be one of {DUPLICATE_SELECTIONS}, got {selection!r}")
    keys = [EMPLOYEE_COL, MONTH_COL, "gross_pay", "total_deductions", "net_pay"]
    valid = df[keys].notna().all(axis=1)
    sub = df[valid]
    group_size = sub.groupby(keys, sort=False)[EMPLOYEE_COL].transform("size")
    if selection == "all":
        flag = group_size > 1
    elif selection == "later_by_item_id":
        # rank by item id (not by row position), so shuffling rows cannot change the result
        rank = sub.groupby(keys, sort=False)[ID_COL].rank(method="first")
        flag = (group_size > 1) & (rank > 1)
    else:  # "file_order" - original experiment
        flag = (group_size > 1) & (sub.groupby(keys, sort=False).cumcount() > 0)
    out = pd.Series(0, index=df.index, dtype=int)
    out.loc[sub.index] = flag.astype(int)
    return out


def previous_month_flag(df: pd.DataFrame, flag: pd.Series) -> pd.Series:
    """True where the same employee had a flagged payslip in the previous CALENDAR month."""
    period = month_period(df)
    per_month = pd.DataFrame({EMPLOYEE_COL: df[EMPLOYEE_COL], "_p": period, "f": flag.astype(bool)}) \
        .groupby([EMPLOYEE_COL, "_p"])["f"].any()
    prev_idx = pd.MultiIndex.from_arrays([df[EMPLOYEE_COL], period - 1])
    return pd.Series(per_month.reindex(prev_idx, fill_value=False).to_numpy(dtype=bool), index=df.index)


def apply_rules(df: pd.DataFrame, cfg: RuleConfig | None = None) -> pd.DataFrame:
    """Return one 0/1 column per rule (see RULE_COLUMNS). NaN inputs never trigger a value rule."""
    cfg = cfg or RuleConfig()
    net, gross = df["net_pay"], df["gross_pay"]
    ot_hours = df["overtime_hours_paid"]

    gross_gap = (gross - df["expected_gross"]).abs() / df["expected_gross"].where(df["expected_gross"] > 0)
    # expected_gross == 0 but gross != 0 is also inconsistent
    gross_zero_conflict = (df["expected_gross"] == 0) & (gross.abs() > 0)

    prev_pf = df["prev_pf_employee"]
    required_numeric = MONEY_COLUMNS + ["payable_days", "lop_days", "overtime_hours_paid"]
    invalid = df[required_numeric].isna().any(axis=1) | ~np.isfinite(df[required_numeric]).all(axis=1)

    flags = pd.DataFrame(index=df.index)
    flags["rule_duplicate_payment"] = detect_duplicate_payments(df, cfg.duplicate_selection)
    flags["rule_negative_net"] = (net < 0).astype(int)
    flags["rule_zero_net"] = ((net == 0) & (df["payable_days"] > 0)).astype(int)
    flags["rule_gross_mismatch"] = ((gross_gap > cfg.gross_tolerance) | gross_zero_conflict).astype(int)
    flags["rule_overtime_cap"] = (
        (ot_hours > cfg.overtime_cap_hours) | ((ot_hours > 0) & (df["overtime_eligible"] != "Y"))
    ).astype(int)
    flags["rule_missed_pf"] = ((df["pf_employee"] == 0) & (gross > 0) & (prev_pf > 0)).astype(int)
    drop = df["fixed_gross_change_vs_prev"] < cfg.fixed_gross_drop_limit
    jump = df["fixed_gross_change_vs_prev"] > cfg.fixed_gross_jump_limit
    role_out = df["fixed_gross_vs_role_median"] > cfg.role_ratio_limit
    if cfg.suppress_drop_after_flagged_spike:       # post-evaluation experiment, off by default
        spike_last_month = previous_month_flag(df, jump | role_out)
        back_to_pre_spike = ((df["monthly_fixed_gross"] / df["prev2_monthly_fixed_gross"].where(df["prev2_monthly_fixed_gross"] != 0)) - 1).abs() <= cfg.reversion_tolerance
        drop = drop & ~(spike_last_month & back_to_pre_spike)
    flags["rule_salary_drop"] = drop.astype(int)
    flags["rule_fixed_gross_jump"] = jump.astype(int)
    flags["rule_role_pay_outlier"] = role_out.astype(int)
    flags["rule_invalid_amounts"] = invalid.astype(int)
    return flags[RULE_COLUMNS]
