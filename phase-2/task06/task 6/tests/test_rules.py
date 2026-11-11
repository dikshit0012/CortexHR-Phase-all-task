import numpy as np
import pandas as pd
import pytest

from conftest import payslip
from cortexhr_task6.payroll_features import build_payroll_features
from cortexhr_task6.payroll_rules import RULE_COLUMNS, RuleConfig, apply_rules, detect_duplicate_payments


def rules_for(rows, cfg=None):
    f = build_payroll_features(pd.DataFrame(rows))
    return apply_rules(f, cfg).set_index(f["payroll_item_id"])


def test_negative_net_is_flagged_separately_from_zero_net():
    r = rules_for([payslip("NEG", net_pay=-10.0), payslip("ZERO", emp="E2", gross_pay=0.0, net_pay=0.0, total_deductions=0.0),
                   payslip("OK", emp="E3")])
    assert r.loc["NEG", "rule_negative_net"] == 1 and r.loc["NEG", "rule_zero_net"] == 0
    assert r.loc["ZERO", "rule_zero_net"] == 1 and r.loc["ZERO", "rule_negative_net"] == 0
    assert r.loc["OK", ["rule_negative_net", "rule_zero_net"]].sum() == 0


def test_zero_net_with_no_payable_days_is_not_flagged():
    r = rules_for([payslip("UNPAID", payable_days=0.0, gross_pay=0.0, net_pay=0.0, total_deductions=0.0)])
    assert r.loc["UNPAID", "rule_zero_net"] == 0       # legitimate: fully unpaid month


def test_duplicate_requires_same_employee_month_and_amounts():
    rows = [payslip("A1", emp="E1", month="2025-04-01"),
            payslip("A2", emp="E1", month="2025-04-01"),                       # true repeat
            payslip("B1", emp="E2", month="2025-04-01"),                       # same amounts, different employee
            payslip("C1", emp="E1", month="2025-05-01"),                       # same amounts, different month
            payslip("D1", emp="E4", month="2025-04-01"),
            payslip("D2", emp="E4", month="2025-04-01", tds=9000.0)]           # same emp+month, different amounts
    rows[-1]["total_deductions"] += 6000.0
    rows[-1]["net_pay"] -= 6000.0
    r = rules_for(rows)["rule_duplicate_payment"]
    # default flags the WHOLE group (the data cannot say which copy is the extra one)
    assert r.to_dict() == {"A1": 1, "A2": 1, "B1": 0, "C1": 0, "D1": 0, "D2": 0}


def test_equal_salary_alone_is_never_a_duplicate():
    f = build_payroll_features(pd.DataFrame([payslip("X1", emp="E1"), payslip("X2", emp="E2")]))
    assert detect_duplicate_payments(f).sum() == 0


def test_duplicate_selection_options():
    f = build_payroll_features(pd.DataFrame([payslip("A2"), payslip("A1")]))     # note: A2 appears first in the file
    assert detect_duplicate_payments(f, "all").tolist() == [1, 1]
    assert detect_duplicate_payments(f, "later_by_item_id").tolist() == [1, 0]    # A2 is the later id, whatever the row order
    assert detect_duplicate_payments(f, "file_order").tolist() == [0, 1]          # original experiment: depends on row position
    with pytest.raises(ValueError, match="duplicate_selection"):
        detect_duplicate_payments(f, "bogus")


def test_gross_mismatch_catches_shifted_decimal_and_ignores_rounding():
    ok = payslip("OK")
    shifted = payslip("TEN", emp="E2")
    shifted["gross_pay"] *= 10
    rounded = payslip("RND", emp="E3")
    rounded["gross_pay"] += 0.01
    r = rules_for([ok, shifted, rounded])["rule_gross_mismatch"]
    assert r.to_dict() == {"OK": 0, "TEN": 1, "RND": 0}


def test_overtime_cap_and_eligibility():
    r = rules_for([payslip("CAP", ot_hours=60.0, ot_eligible="Y"), payslip("OKOT", emp="E2", ot_hours=10.0, ot_eligible="Y"),
                   payslip("NOELIG", emp="E3", ot_hours=5.0, ot_eligible="N")])["rule_overtime_cap"]
    assert r.to_dict() == {"CAP": 1, "OKOT": 0, "NOELIG": 1}


def test_missed_pf_needs_history_of_pf(base_frame):
    df = base_frame.copy()
    df.loc[(df.employee_id == "E1") & (df.month == "2025-06-01"), "pf_employee"] = 0.0          # stopped after paying
    first_month = (df.employee_id == "E2") & (df.month == "2025-04-01")
    df.loc[first_month, "pf_employee"] = 0.0                                                    # no history
    df.loc[df.employee_id == "E3", "pf_employee"] = 0.0                                         # always exempt
    f = build_payroll_features(df)
    r = apply_rules(f).set_index(f["payroll_item_id"])["rule_missed_pf"]
    flagged = set(r.index[r == 1])
    assert flagged == set(df.loc[(df.employee_id == "E1") & (df.month == "2025-06-01"), "payroll_item_id"])


def test_role_outlier_jump_and_drop_rules(base_frame):
    df = base_frame.copy()
    df.loc[(df.employee_id == "E1") & (df.month == "2025-06-01"), "monthly_fixed_gross"] = 160000.0   # 3x jump, 3x role median
    f = build_payroll_features(df)
    r = apply_rules(f).set_index([f.employee_id, f.month.dt.strftime("%Y-%m")])
    assert r.loc[("E1", "2025-06"), "rule_fixed_gross_jump"] == 1
    assert r.loc[("E1", "2025-06"), "rule_role_pay_outlier"] == 1
    assert r.loc[("E1", "2025-07"), "rule_salary_drop"] == 1         # return to normal looks like a drop (documented false positive)
    assert r.loc[("E2", "2025-06"), RULE_COLUMNS].sum() == 0


def test_invalid_amount_is_flagged(make_frame):
    r = rules_for([payslip("BAD", tds=np.nan), payslip("OK", emp="E2")])["rule_invalid_amounts"]
    assert r.to_dict() == {"BAD": 1, "OK": 0}


def test_clean_data_triggers_no_rule(base_frame):
    f = build_payroll_features(base_frame)
    assert apply_rules(f).to_numpy().sum() == 0


def test_rules_are_configurable():
    rows = [payslip("CAP", ot_hours=60.0, ot_eligible="Y")]
    assert rules_for(rows, RuleConfig(overtime_cap_hours=100.0)).loc["CAP", "rule_overtime_cap"] == 0
