"""Reason mapping: every detector signal has a code, every code has a readable sentence, primary reason is deterministic."""
import numpy as np
import pandas as pd
import pytest

from conftest import attendance_row, payslip, stub_flags
from cortexhr_task5.attendance_anomaly_detector import RULE_COLUMNS as T5_RULES
from cortexhr_task5.attendance_anomaly_detector import run_detector
from cortexhr_task6.payroll_features import build_payroll_features
from cortexhr_task6.payroll_rules import RULE_COLUMNS as T6_RULES
from cortexhr_task7.explain import ATTENDANCE_RULE_MAP, PAYROLL_RULE_MAP, explain_attendance, explain_payroll
from cortexhr_task7.reason_codes import PRIORITY, REASON_CODES, reason_code_dictionary
from cortexhr_task7.reason_text import BANNED_TERMS, TEXT_BUILDERS, _about, _n, build_reason_text

RECORD = dict(copies=2, period=pd.Timestamp("2025-06-01"), dup_total=2000.0, money=1000.0, net_pay=-500.0,
              total_deductions=9000.0, gross_pay=8500.0, payable_days=30.0, expected_gross=8400.0,
              overtime_hours_paid=60.0, overtime_cap=50.0, overtime_pay=3000.0, prev_pf_employee=1800.0,
              monthly_fixed_gross=90000.0, role_median_fixed_gross=40000.0, prev_monthly_fixed_gross=45000.0,
              prev_month_spike=False, scheduled_days=22.0, longest_unpaid_streak_days=8.0, avg_hours_worked=15.5)


def one_row_feats(**over):
    return build_payroll_features(pd.DataFrame([payslip("P1", **over)]))


# ---------------------------------------------------------------- inventory: every detector signal is covered
def test_every_task6_rule_is_mapped_to_reason_codes():
    assert set(T6_RULES) == set(PAYROLL_RULE_MAP)
    for rule, codes in PAYROLL_RULE_MAP.items():
        assert codes and all(c in REASON_CODES for c in codes), rule


def test_every_task5_rule_is_mapped_to_reason_codes():
    assert set(T5_RULES) == set(ATTENDANCE_RULE_MAP)
    for rule, codes in ATTENDANCE_RULE_MAP.items():
        assert codes and all(c in REASON_CODES for c in codes), rule


def test_model_only_signals_have_codes():
    assert REASON_CODES["STATISTICAL_OUTLIER"].source == "payroll" and REASON_CODES["PATTERN_SHIFT"].source == "attendance"


def test_dictionary_is_complete_and_consistent():
    assert PRIORITY == list(REASON_CODES) and len(set(PRIORITY)) == len(PRIORITY)
    assert set(TEXT_BUILDERS) == set(REASON_CODES)
    d = reason_code_dictionary()
    assert list(d["reason_code"]) == PRIORITY
    for col in ("meaning", "money_at_risk_basis", "detector_signal"):
        assert d[col].astype(str).str.len().gt(5).all()


# ---------------------------------------------------------------- readable text
@pytest.mark.parametrize("code", PRIORITY)
def test_reason_text_is_nonempty_plain_and_ends_with_a_next_step(code):
    text = build_reason_text(code, dict(RECORD))
    low = text.lower()
    assert text.strip() and "nan" not in low.split()
    assert "please" in low, "every explanation must say what to check next"
    assert not [t for t in BANNED_TERMS if t in low], f"technical wording in {code}: {text}"
    assert len(text.split()) <= 70


def test_missing_numbers_never_print_as_nan():
    rec = dict(RECORD, money=np.nan, net_pay=np.nan, expected_gross=np.nan, prev_pf_employee=np.nan)
    for code in PRIORITY:
        assert "nan" not in build_reason_text(code, rec).lower().replace("financial", "")
    assert _n(np.nan) == "an unknown amount" and _about(np.nan) == ""


def test_unknown_reason_code_is_an_error_not_a_blank():
    with pytest.raises(KeyError, match="no reason text"):
        build_reason_text("NOT_A_CODE", dict(RECORD))


def test_model_only_flags_get_a_cautious_explanation_without_an_amount():
    feats = build_payroll_features(pd.DataFrame([payslip("P1")]))
    out = explain_payroll(feats, stub_flags(feats, model_flags=[1]))
    row = out.iloc[0]
    assert row.primary_reason_code == "STATISTICAL_OUTLIER" and pd.isna(row.money_at_risk)
    low = row.reason_text.lower()
    assert "not known" in low and "no amount" in low and "no specific payroll check failed" in low
    for claim in ("error", "mistake", "lost", "fraud", "wrong"):
        assert claim not in low


def test_attendance_model_only_flag_is_cautious(clean_attendance):
    feats, flags, _, _ = run_detector(clean_attendance)
    flags = flags.copy()
    flags["if_flag"], flags["combined_anomaly"] = 0, 0
    flags.iloc[0, flags.columns.get_loc("if_flag")] = 1
    flags.iloc[0, flags.columns.get_loc("combined_anomaly")] = 1
    out = explain_attendance(feats, flags)
    assert len(out) == 1 and out.iloc[0].primary_reason_code == "PATTERN_SHIFT"
    assert "not known" in out.iloc[0].reason_text and pd.isna(out.iloc[0].money_at_risk)


# ---------------------------------------------------------------- primary reason
def test_primary_reason_follows_the_documented_priority_when_several_signals_fire():
    # zero pay also breaks the gross check and the model flags it too: ZERO_PAY must win. The model's agreement is kept
    # as extra evidence (model_flag, signal_count), not as a reason, because STATISTICAL_OUTLIER means "model ONLY".
    feats = build_payroll_features(pd.DataFrame([payslip("P1", gross_pay=0.0, net_pay=0.0, total_deductions=0.0)]))
    out = explain_payroll(feats, stub_flags(feats, model_flags=[1]))
    row = out.iloc[0]
    assert row.primary_reason_code == "ZERO_PAY"
    assert row.all_reason_codes == "ZERO_PAY;GROSS_BELOW_EXPECTED"
    assert row.model_flag == 1 and row.rule_count == 2 and row.signal_count == 3


def test_primary_reason_is_the_same_whatever_the_row_order():
    rows = [payslip("P1", gross_pay=0.0, net_pay=0.0, total_deductions=0.0), payslip("P2", emp="E2", net_pay=-10.0),
            payslip("P3", emp="E3", ot_hours=70.0, ot_eligible="Y")]
    results = []
    for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
        feats = build_payroll_features(pd.DataFrame([rows[i] for i in order]))
        out = explain_payroll(feats, stub_flags(feats))
        results.append(dict(zip(out.record_id, out.primary_reason_code)))
    assert results[0] == results[1] == results[2]


def test_gross_direction_selects_above_or_below_code():
    above = one_row_feats(gross_pay=60000.0)
    below = one_row_feats(gross_pay=40000.0)
    assert explain_payroll(above, stub_flags(above)).iloc[0].primary_reason_code == "GROSS_ABOVE_NORM"
    assert explain_payroll(below, stub_flags(below)).iloc[0].primary_reason_code == "GROSS_BELOW_EXPECTED"


def test_overtime_cap_versus_not_eligible():
    over = one_row_feats(ot_hours=60.0, ot_eligible="Y")
    not_eligible = one_row_feats(ot_hours=5.0, ot_eligible="N")
    assert explain_payroll(over, stub_flags(over)).iloc[0].primary_reason_code == "OVERTIME_ABOVE_CAP"
    assert explain_payroll(not_eligible, stub_flags(not_eligible)).iloc[0].primary_reason_code == "OVERTIME_NOT_ELIGIBLE"


@pytest.mark.parametrize("kwargs,code", [
    ({"net_pay": -5.0}, "NEGATIVE_NET_PAY"),
    ({"gross_pay": 0.0, "net_pay": 0.0, "total_deductions": 0.0}, "ZERO_PAY"),
    ({"tds": np.nan}, "INVALID_PAYROLL_DATA"),
])
def test_single_payroll_rules_produce_their_code(kwargs, code):
    feats = one_row_feats(**kwargs)
    assert explain_payroll(feats, stub_flags(feats)).iloc[0].primary_reason_code == code


# ---------------------------------------------------------------- unknown signals are explicit
def test_unmapped_rule_column_in_flags_raises():
    feats = one_row_feats(net_pay=-5.0)
    flags = stub_flags(feats)
    flags["rule_brand_new_check"] = 1
    with pytest.raises(ValueError, match="out of date"):
        explain_payroll(feats, flags)


def test_missing_rule_column_in_flags_raises():
    feats = one_row_feats(net_pay=-5.0)
    flags = stub_flags(feats).drop(columns=["rule_missed_pf"])
    with pytest.raises(ValueError, match="out of date"):
        explain_payroll(feats, flags)


def test_flag_without_any_explainable_signal_raises_instead_of_being_dropped():
    feats = one_row_feats()
    flags = stub_flags(feats)
    flags["combined_flag"] = 1                           # flagged, but no rule fired and the model did not flag it
    with pytest.raises(ValueError, match="no explainable signal"):
        explain_payroll(feats, flags)


def test_attendance_split_must_reproduce_task5_rules(clean_attendance):
    df = clean_attendance.copy()
    df.loc[0, "longest_unpaid_streak_days"] = 8.0
    feats, flags, _, _ = run_detector(df)
    tampered = flags.copy()
    tampered["rule_unexcused_absence"] = 0               # pretend Task 5 rules changed without updating the mapping
    with pytest.raises(ValueError, match="no longer matches"):
        explain_attendance(feats, tampered)


def test_attendance_rules_map_to_the_right_codes(clean_attendance):
    df = clean_attendance.copy()
    df.loc[0, ["present_days", "paid_leave_days"]] = [0.0, 0.0]
    df.loc[1, "longest_unpaid_streak_days"] = 8.0
    df.loc[2, "avg_hours_worked"] = 16.0
    feats, flags, _, _ = run_detector(df)
    out = explain_attendance(feats, flags).set_index("record_id")
    assert out.loc["A0001", "primary_reason_code"] == "NO_ATTENDANCE"
    assert out.loc["A0002", "primary_reason_code"] == "LONG_UNPAID_ABSENCE"
    assert out.loc["A0003", "primary_reason_code"] == "IMPLAUSIBLE_HOURS"
    assert "8 unpaid absence days" in out.loc["A0002", "reason_text"] and "16.0 hours" in out.loc["A0003", "reason_text"]
