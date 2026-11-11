"""Task 7 - `money_at_risk` estimates for payroll flags (amounts in INR, as in the dataset).

What the number means
    An ESTIMATE of the money that may be affected IF the flagged payslip is wrong: an overpayment, an
    underpayment or an over-deduction. It is exposure to be verified, NOT confirmed loss, and an unusual
    payslip is not proof that the company lost this amount.

Rules for every formula
    * Result is >= 0, rounded to 2 decimals. Missing, zero-denominator or non-finite inputs give NaN
      (unknown), never an invented number and never 0 to mean "unknown".
    * Only the PRIMARY reason's formula is shown for a record, so overlapping signals are never added
      together. The other candidate amounts are kept in the detail report.
    * Statistical-model-only flags and attendance flags get NaN: no defensible amount exists.
    * No evaluation labels are used anywhere in this module.

Reference values (label-free)
    usual deduction share = the employee's median (total deductions / gross pay) over PREVIOUS calendar
    months; if there is no history, the dataset-wide median. Using only past months keeps it available at
    the time the payslip is reviewed.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from cortexhr_task6.payroll_features import EMPLOYEE_COL, MONTH_COL, month_period

DUP_KEYS = [EMPLOYEE_COL, MONTH_COL, "gross_pay", "total_deductions", "net_pay"]   # same keys as the Task 6 rule


def clean_amount(values) -> pd.Series:
    """Non-negative, 2-decimal, finite amounts; anything else becomes NaN."""
    s = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s.clip(lower=0).round(2)


def prior_median_deduction_share(feats: pd.DataFrame) -> Tuple[pd.Series, float]:
    """(per-row usual deduction share from PREVIOUS months, dataset-wide fallback).

    Order independent: months are aggregated per employee first, then a shifted expanding median is used,
    so a repeated payslip never becomes its own history.
    """
    period = month_period(feats)
    per_month = (feats.assign(_p=period).groupby([EMPLOYEE_COL, "_p"], as_index=False)["deduction_ratio"].median()
                 .sort_values([EMPLOYEE_COL, "_p"]))
    per_month["_prior"] = per_month.groupby(EMPLOYEE_COL)["deduction_ratio"].transform(
        lambda s: s.shift(1).expanding().median())
    key = feats[[EMPLOYEE_COL]].assign(_p=period)
    prior = key.merge(per_month[[EMPLOYEE_COL, "_p", "_prior"]], on=[EMPLOYEE_COL, "_p"], how="left")["_prior"]
    prior.index = feats.index
    global_share = float(feats["deduction_ratio"].median())
    return prior, global_share


def usual_deduction_share(feats: pd.DataFrame) -> pd.Series:
    prior, global_share = prior_median_deduction_share(feats)
    share = prior.fillna(global_share)
    return share.where((share >= 0) & (share < 1))       # a share outside [0, 1) is not usable -> NaN


def _prorate(feats: pd.DataFrame) -> pd.Series:
    """payable_days / days in month (the part of the month that was paid)."""
    dim = feats[MONTH_COL].dt.days_in_month
    return (feats["payable_days"] / dim).where(dim > 0)


def duplicate_amounts(feats: pd.DataFrame, flag: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """(row share, group total) of the possible extra payment for rows flagged as duplicates.

    group total = (copies - 1) x net pay;  each flagged copy carries total / (flagged copies in the group).
    The shares add up to the group total, so nothing is counted twice, and the split needs no choice of
    'which copy is the extra one' (the data has no payment reference to decide that).
    """
    sub = feats[DUP_KEYS].notna().all(axis=1)
    share = pd.Series(np.nan, index=feats.index)
    total = pd.Series(np.nan, index=feats.index)
    if sub.any():
        g = feats[sub].assign(_flag=flag[sub].astype(int)).groupby(DUP_KEYS, sort=False)
        copies = g[EMPLOYEE_COL].transform("size")
        flagged = g["_flag"].transform("sum")
        net = feats.loc[sub, "net_pay"].clip(lower=0)
        grp_total = (copies - 1) * net
        total.loc[sub] = grp_total
        share.loc[sub] = (grp_total / flagged.where(flagged > 0))
    return clean_amount(share).set_axis(feats.index), clean_amount(total).set_axis(feats.index)


def candidate_amounts(feats: pd.DataFrame, flags: pd.DataFrame, overtime_cap: float = 50.0) -> pd.DataFrame:
    """One candidate amount column per payroll reason code (NaN where the formula cannot be evaluated)."""
    gross, net = feats["gross_pay"], feats["net_pay"]
    expected = feats["expected_gross"]
    prorate = _prorate(feats)
    share = usual_deduction_share(feats)
    dup_share, _ = duplicate_amounts(feats, flags["rule_duplicate_payment"])

    out = pd.DataFrame(index=feats.index)
    out["DUPLICATE_PAYMENT"] = dup_share

    # negative net pay: deductions above the employee's usual share of gross (never below |net|, which is the minimum error)
    usual_ded = share * gross.clip(lower=0)
    excess_ded = feats["total_deductions"] - usual_ded
    min_error = (-net).where(net < 0)                       # |net|: deductions are wrong by at least this much
    out["NEGATIVE_NET_PAY"] = clean_amount(pd.concat([excess_ded, min_error], axis=1).max(axis=1, skipna=True))

    # zero net pay: net pay that should have been paid
    exp_net = expected.where(expected > 0) * (1 - share)
    out["ZERO_PAY"] = clean_amount(exp_net - net.clip(lower=0))

    diff = (gross - expected).abs()
    out["GROSS_ABOVE_NORM"] = clean_amount(diff.where(gross > expected))
    out["GROSS_BELOW_EXPECTED"] = clean_amount(diff.where(gross < expected))

    hours, ot_pay = feats["overtime_hours_paid"], feats["overtime_pay"]
    out["OVERTIME_ABOVE_CAP"] = clean_amount(ot_pay * (hours - overtime_cap) / hours.where(hours > 0))
    out["OVERTIME_NOT_ELIGIBLE"] = clean_amount(ot_pay)
    out["PF_NOT_DEDUCTED"] = clean_amount(feats["prev_pf_employee"].where(feats["prev_pf_employee"] > 0))
    out["PAY_ABOVE_ROLE_NORM"] = clean_amount((feats["monthly_fixed_gross"] - feats["role_median_fixed_gross"]) * prorate)
    out["SALARY_JUMP"] = clean_amount((feats["monthly_fixed_gross"] - feats["prev_monthly_fixed_gross"]) * prorate)
    out["SALARY_DROP"] = clean_amount((feats["prev_monthly_fixed_gross"] - feats["monthly_fixed_gross"]) * prorate)
    out["INVALID_PAYROLL_DATA"] = np.nan
    out["STATISTICAL_OUTLIER"] = np.nan
    return out
