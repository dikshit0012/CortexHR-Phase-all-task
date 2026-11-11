"""money_at_risk: every formula checked against a hand-computed number (INR), plus the safety properties."""
import numpy as np
import pandas as pd
import pytest

from conftest import payslip, stub_flags
from cortexhr_task6.detection import run_detection
from cortexhr_task6.payroll_features import build_payroll_features
from cortexhr_task6.payroll_rules import RuleConfig
from cortexhr_task7.explain import explain_payroll
from cortexhr_task7.money_at_risk import candidate_amounts, clean_amount, duplicate_amounts, usual_deduction_share


def explain(rows, model_flags=None, cfg=None):
    feats = build_payroll_features(pd.DataFrame(rows))
    flags = stub_flags(feats, model_flags, cfg)
    return feats, flags, explain_payroll(feats, flags, cfg).set_index("record_id")


def history(emp="E1", months=("2025-02-01", "2025-03-01"), **kw):
    return [payslip(f"{emp}-{m[5:7]}", emp=emp, month=m, **kw) for m in months]


# ---------------------------------------------------------------- gross pay mismatch
def test_gross_above_expected_is_the_difference():
    _, _, out = explain([payslip("P1", gross_pay=55000.0)])               # expected gross is 50,000
    assert out.loc["P1", "primary_reason_code"] == "GROSS_ABOVE_NORM" and out.loc["P1", "money_at_risk"] == 5000.00


def test_gross_below_expected_is_the_difference():
    _, _, out = explain([payslip("P1", gross_pay=44000.0)])
    assert out.loc["P1", "primary_reason_code"] == "GROSS_BELOW_EXPECTED" and out.loc["P1", "money_at_risk"] == 6000.00


def test_gross_difference_uses_overtime_and_variable_pay_in_the_expected_amount():
    # expected = 50,000 + variable 10,000 = 60,000; recorded 66,000 -> 6,000
    _, _, out = explain([payslip("P1", variable=10000.0, gross_pay=66000.0)])
    assert out.loc["P1", "money_at_risk"] == 6000.00


# ---------------------------------------------------------------- duplicate payments: no double counting
def test_duplicate_pair_is_split_so_the_total_is_one_extra_payment():
    a, b = payslip("A1", emp="E1"), payslip("A2", emp="E1")
    _, _, out = explain([a, b])
    net = a["net_pay"]
    assert out.loc["A1", "money_at_risk"] == out.loc["A2", "money_at_risk"] == round(net / 2, 2)
    assert out["money_at_risk"].sum() == pytest.approx(net, abs=0.01)             # one extra net payment, counted once


def test_three_identical_payslips_mean_two_extra_payments():
    rows = [payslip(f"A{i}", emp="E1") for i in range(1, 4)]
    _, _, out = explain(rows)
    assert out["money_at_risk"].sum() == pytest.approx(2 * rows[0]["net_pay"], abs=0.02)


def test_duplicate_amount_is_not_the_whole_payslip_of_every_copy():
    rows = [payslip("A1", emp="E1"), payslip("A2", emp="E1")]
    _, _, out = explain(rows)
    assert out["money_at_risk"].sum() < 2 * rows[0]["net_pay"]


def test_only_the_flagged_copies_carry_the_amount():
    rows = [payslip("A1", emp="E1"), payslip("A2", emp="E1")]
    _, _, out = explain(rows, cfg=RuleConfig(duplicate_selection="later_by_item_id"))
    assert list(out.index) == ["A2"] and out.loc["A2", "money_at_risk"] == round(rows[0]["net_pay"], 2)


def test_same_salary_for_different_employees_is_not_a_duplicate():
    feats, flags, _ = explain([payslip("A1", emp="E1"), payslip("A2", emp="E2")])
    assert flags["rule_duplicate_payment"].sum() == 0
    assert duplicate_amounts(feats, flags["rule_duplicate_payment"])[0].isna().all()          # no flagged copy -> no amount


# ---------------------------------------------------------------- negative / zero net pay
def test_negative_net_uses_the_employees_usual_deduction_share():
    rows = history() + [payslip("E1-04", month="2025-04-01", total_deductions=80000.0, net_pay=-30000.0)]
    _, _, out = explain(rows)
    # usual share 5,000 / 50,000 = 10%; usual deductions 5,000; excess 75,000 (NOT abs(net) = 30,000)
    assert out.loc["E1-04", "primary_reason_code"] == "NEGATIVE_NET_PAY"
    assert out.loc["E1-04", "money_at_risk"] == 75000.00


def test_negative_net_without_history_uses_the_dataset_median_share():
    rows = history("E1") + [payslip("E2-01", emp="E2", month="2025-04-01", total_deductions=80000.0, net_pay=-30000.0)]
    _, _, out = explain(rows)
    assert out.loc["E2-01", "money_at_risk"] == 75000.00                           # dataset median share is also 10%


def test_negative_net_falls_back_to_the_minimum_error_when_no_share_is_usable():
    _, _, out = explain([payslip("P1", total_deductions=80000.0, net_pay=-30000.0)])   # only row: its own share (160%) is unusable
    assert out.loc["P1", "money_at_risk"] == 30000.00                              # deductions are wrong by AT LEAST |net|


def test_zero_net_is_expected_net_pay():
    rows = history() + [payslip("E1-04", month="2025-04-01", gross_pay=0.0, net_pay=0.0, total_deductions=0.0)]
    _, _, out = explain(rows)
    assert out.loc["E1-04", "primary_reason_code"] == "ZERO_PAY"
    assert out.loc["E1-04", "money_at_risk"] == 45000.00                           # 50,000 x (1 - 10%)


def test_zero_net_without_expected_gross_has_no_estimate():
    rows = history() + [payslip("E1-04", month="2025-04-01", gross_pay=0.0, net_pay=0.0, total_deductions=0.0, variable_pay=np.nan)]
    _, _, out = explain(rows)
    assert out.loc["E1-04", "primary_reason_code"] == "ZERO_PAY" and pd.isna(out.loc["E1-04", "money_at_risk"])
    assert "INVALID_PAYROLL_DATA" in out.loc["E1-04", "all_reason_codes"]


# ---------------------------------------------------------------- overtime, PF, history, role
def test_overtime_above_cap_is_the_pay_for_the_excess_hours():
    p = payslip("P1", ot_hours=60.0, ot_eligible="Y")
    _, _, out = explain([p])
    assert out.loc["P1", "primary_reason_code"] == "OVERTIME_ABOVE_CAP"
    assert out.loc["P1", "money_at_risk"] == round(p["overtime_pay"] * 10 / 60, 2)


def test_overtime_for_non_eligible_role_is_all_overtime_pay():
    p = payslip("P1", ot_hours=5.0, ot_eligible="N")
    _, _, out = explain([p])
    assert out.loc["P1", "primary_reason_code"] == "OVERTIME_NOT_ELIGIBLE" and out.loc["P1", "money_at_risk"] == round(p["overtime_pay"], 2)


def test_missed_pf_is_last_months_pf_and_needs_history():
    rows = history() + [payslip("E1-04", month="2025-04-01", pf_employee=0.0, total_deductions=3200.0, net_pay=46800.0)]
    _, _, out = explain(rows)
    assert out.loc["E1-04", "primary_reason_code"] == "PF_NOT_DEDUCTED" and out.loc["E1-04", "money_at_risk"] == 1800.00


def test_salary_drop_is_prorated_by_days_paid():
    rows = history(months=("2025-02-01", "2025-03-01")) + [payslip("E1-04", month="2025-04-01", fixed=40000.0, payable_days=15.0)]
    _, _, out = explain(rows)
    assert out.loc["E1-04", "primary_reason_code"] == "SALARY_DROP"
    assert out.loc["E1-04", "money_at_risk"] == 5000.00                            # (50,000 - 40,000) x 15/30


def test_salary_jump_is_the_extra_fixed_pay():
    rows = history() + [payslip("E1-04", month="2025-04-01", fixed=90000.0)]
    _, _, out = explain(rows)
    assert out.loc["E1-04", "primary_reason_code"] == "SALARY_JUMP" and out.loc["E1-04", "money_at_risk"] == 40000.00


def test_pay_above_role_norm_is_the_excess_over_the_role_median():
    rows = []
    for emp in ("E2", "E3", "E4"):
        rows += history(emp)
    rows += history("E1") + [payslip("E1-04", emp="E1", month="2025-04-01", fixed=130000.0)]
    _, _, out = explain(rows)
    assert out.loc["E1-04", "primary_reason_code"] == "PAY_ABOVE_ROLE_NORM"           # outranks SALARY_JUMP (also fired)
    assert "SALARY_JUMP" in out.loc["E1-04", "all_reason_codes"]
    assert out.loc["E1-04", "money_at_risk"] == 80000.00                              # (130,000 - role median 50,000)


# ---------------------------------------------------------------- no unsupported estimates, no double counting
def test_model_only_and_invalid_data_flags_get_no_amount():
    _, _, out = explain([payslip("P1"), payslip("P2", emp="E2", tds=np.nan)], model_flags=[1, 0])
    assert pd.isna(out.loc["P1", "money_at_risk"]) and out.loc["P1", "primary_reason_code"] == "STATISTICAL_OUTLIER"
    assert out.loc["P2", "primary_reason_code"] == "INVALID_PAYROLL_DATA" and pd.isna(out.loc["P2", "money_at_risk"])


def test_overlapping_signals_show_one_amount_not_the_sum():
    rows = history() + [payslip("E1-04", month="2025-04-01", gross_pay=0.0, net_pay=0.0, total_deductions=0.0)]
    feats, flags, out = explain(rows)
    row = out.loc["E1-04"]
    cands = candidate_amounts(feats, flags).loc[feats.index[feats.payroll_item_id == "E1-04"][0]]
    assert row.primary_reason_code == "ZERO_PAY" and "GROSS_BELOW_EXPECTED" in row.all_reason_codes
    assert row.money_at_risk == cands["ZERO_PAY"]
    assert row.money_at_risk < cands["ZERO_PAY"] + cands["GROSS_BELOW_EXPECTED"]
    assert "GROSS_BELOW_EXPECTED=" in row.candidate_amounts and "ZERO_PAY=" in row.candidate_amounts   # alternatives stay visible


# ---------------------------------------------------------------- bad numbers
def test_zero_denominators_give_nan_not_infinity():
    feats = build_payroll_features(pd.DataFrame([payslip("P1", ot_hours=0.0, gross_pay=0.0, net_pay=0.0, total_deductions=0.0)]))
    cands = candidate_amounts(feats, stub_flags(feats))
    assert not np.isinf(cands.to_numpy(dtype=float)).any()
    assert pd.isna(cands["OVERTIME_ABOVE_CAP"].iloc[0])                              # hours = 0 -> undefined, not inf


def test_invalid_and_infinite_inputs_do_not_crash_and_give_no_amount():
    rows = [payslip("P1", gross_pay=np.inf), payslip("P2", emp="E2", net_pay="oops"), payslip("P3", emp="E3", payable_days=np.nan)]
    feats, flags, out = explain(rows)
    cands = candidate_amounts(feats, flags)
    assert not np.isinf(cands.to_numpy(dtype=float)).any()
    assert set(out["primary_reason_code"]) == {"INVALID_PAYROLL_DATA"} and out["money_at_risk"].isna().all()


def test_clean_amount_rules():
    s = clean_amount([5.126, -3.0, np.inf, np.nan, 0.0, "x"])
    assert s.iloc[0] == 5.13 and s.iloc[1] == 0.0 and pd.isna(s.iloc[2]) and pd.isna(s.iloc[3]) and s.iloc[4] == 0.0 and pd.isna(s.iloc[5])


def test_usual_share_ignores_unusable_values():
    feats = build_payroll_features(pd.DataFrame([payslip("P1", total_deductions=90000.0, net_pay=-40000.0)]))
    assert usual_deduction_share(feats).isna().all()                                 # a share of 180% is not a usual deduction share


def test_amounts_are_never_negative_or_infinite_on_a_planted_panel(planted_panel):
    df, _ = planted_panel
    feats, flags, _ = run_detection(df)
    out = explain_payroll(feats, flags)
    m = out["money_at_risk"].dropna()
    assert (m >= 0).all() and np.isfinite(m).all()
    cands = candidate_amounts(feats, flags).stack().dropna()
    assert (cands >= 0).all() and np.isfinite(cands).all()


def test_planted_problems_get_the_expected_reason_and_amount(planted_panel):
    df, ids = planted_panel
    feats, flags, _ = run_detection(df)
    out = explain_payroll(feats, flags).set_index("record_id")
    assert out.loc[ids["negative_net"], "primary_reason_code"] == "NEGATIVE_NET_PAY"
    assert out.loc[ids["wrong_decimal"], "primary_reason_code"] == "GROSS_ABOVE_NORM"
    assert out.loc[ids["zero_pay"], "primary_reason_code"] == "ZERO_PAY"
    assert out.loc[ids["overtime"], "primary_reason_code"] == "OVERTIME_ABOVE_CAP"
    assert out.loc[ids["missed_pf"], "primary_reason_code"] == "PF_NOT_DEDUCTED"
    assert {out.loc[ids["duplicate_original"], "primary_reason_code"], out.loc[ids["duplicate_copy"], "primary_reason_code"]} == {"DUPLICATE_PAYMENT"}
    orig = df[df.payroll_item_id == ids["wrong_decimal"]].iloc[0]
    assert out.loc[ids["wrong_decimal"], "money_at_risk"] == pytest.approx(orig["gross_pay"] * 0.9, rel=1e-6, abs=0.01)   # 10x -> 9/10 of it is excess
