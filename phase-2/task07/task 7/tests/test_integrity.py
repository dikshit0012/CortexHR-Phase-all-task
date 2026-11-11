"""Data integrity and no-leakage: right records, nothing dropped or invented, labels never used, identifiers protected."""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cortexhr_task7
from conftest import attendance_row, payslip, stub_flags
from cortexhr_task6.detection import run_detection
from cortexhr_task6.payroll_features import build_payroll_features
from cortexhr_task7 import explain, money_at_risk, official_output, pipeline, ranking, reason_text
from cortexhr_task7.explain import explain_attendance, explain_payroll
from cortexhr_task7.official_output import to_official
from cortexhr_task7.pipeline import assemble, build_official_output, verify_integrity
from cortexhr_task7.review_pack import HUMAN_COLUMNS, build_review_sheet, lint_text, select_sample

PAY, ATT = "payroll_monthly_features", "attendance_monthly_features"


def small_official():
    pay = pd.DataFrame([payslip("P1", net_pay=-5.0), payslip("P2", emp="E2", gross_pay=0.0, net_pay=0.0, total_deductions=0.0)])
    feats = build_payroll_features(pay)
    expl = explain_payroll(feats, stub_flags(feats))
    official, detail = assemble(expl, expl.iloc[0:0])
    return official, detail, pay


# ---------------------------------------------------------------- matching, nothing dropped or invented
def test_integrity_passes_for_a_correct_output():
    official, _, pay = small_official()
    att = pd.DataFrame([attendance_row("A1")])
    verify_integrity(official, pay, att, {PAY: 2, ATT: 0})


def test_a_dropped_flag_is_detected():
    official, _, pay = small_official()
    with pytest.raises(ValueError, match="dropped or invented"):
        verify_integrity(official.iloc[:1], pay, pd.DataFrame([attendance_row("A1")]), {PAY: 2, ATT: 0})


def test_an_invented_record_is_detected():
    official, _, pay = small_official()
    bad = official.copy()
    bad.loc[0, "record_id"] = "P-NOT-IN-SOURCE"
    with pytest.raises(ValueError, match="not found in the source"):
        verify_integrity(bad, pay, pd.DataFrame([attendance_row("A1")]), {PAY: 2, ATT: 0})


def test_a_wrong_employee_or_period_is_detected():
    official, _, pay = small_official()
    att = pd.DataFrame([attendance_row("A1")])
    wrong_emp = official.copy(); wrong_emp.loc[0, "employee_id"] = "E999"
    wrong_period = official.copy(); wrong_period.loc[0, "period"] = "2030-01-01"
    with pytest.raises(ValueError, match="employee_id does not match"):
        verify_integrity(wrong_emp, pay, att, {PAY: 2, ATT: 0})
    with pytest.raises(ValueError, match="period does not match"):
        verify_integrity(wrong_period, pay, att, {PAY: 2, ATT: 0})


def test_misaligned_features_and_flags_are_rejected_not_joined_silently():
    feats = build_payroll_features(pd.DataFrame([payslip("P1", net_pay=-5.0), payslip("P2", emp="E2")]))
    flags = stub_flags(feats).iloc[::-1]
    flags.index = [5, 6]
    with pytest.raises(ValueError, match="not aligned"):
        explain_payroll(feats, flags)
    att = pd.DataFrame([attendance_row("A1")])
    with pytest.raises(ValueError, match="not aligned"):
        explain_attendance(att.set_index(pd.Index([3])), pd.DataFrame({"if_flag": [0]}, index=[9]))


def test_every_flag_from_the_real_detectors_appears_exactly_once(planted_panel, planted_attendance):
    pay, _ = planted_panel
    att, _ = planted_attendance
    official, detail, info = build_official_output(pay, att)
    assert len(official) == sum(info["expected_counts"].values()) and len(official) > 6
    assert official["record_id"].is_unique and official["result_id"].is_unique
    feats6, flags6, _ = run_detection(pay)
    expected = set(feats6.loc[flags6["combined_flag"] == 1, "payroll_item_id"])
    assert set(official.loc[official.source_dataset == PAY, "record_id"]) == expected      # none dropped, none invented
    assert official["reason_code"].notna().all() and (official["reason_text"].str.len() > 20).all()


def test_result_ids_follow_the_ranking_and_are_sequential(planted_panel, planted_attendance):
    official, _, _ = build_official_output(planted_panel[0], planted_attendance[0])
    assert list(official["result_id"]) == [f"AN-{i:04d}" for i in range(1, len(official) + 1)]
    money = official["money_at_risk"]
    n = int(money.notna().sum())
    assert money.iloc[:n].is_monotonic_decreasing and money.iloc[n:].isna().all()


# ---------------------------------------------------------------- no label leakage
def test_no_function_accepts_labels_or_targets():
    for fn in (explain_payroll, explain_attendance, build_official_output, assemble, ranking.rank_flags,
               money_at_risk.candidate_amounts, to_official, select_sample):
        names = set(inspect.signature(fn).parameters)
        assert not any("label" in n or "target" in n or n == "y" for n in names), fn


def test_explanation_modules_never_import_or_mention_the_label_code():
    for module in (explain, money_at_risk, reason_text, ranking, pipeline, official_output):
        src = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in ("evaluation_check", "load_payroll_labels", "anomaly_type", "payroll_anomaly_labels", "unlock_CX"):
            assert forbidden not in src, f"{module.__name__} mentions {forbidden}"


def test_outputs_do_not_change_when_label_like_columns_are_present(planted_panel, planted_attendance):
    pay, _ = planted_panel
    att, _ = planted_attendance
    a, da, _ = build_official_output(pay, att)
    b, db, _ = build_official_output(pay.assign(anomaly_type="DOUBLE_PAYMENT", is_injected=1, injection_note="x"),
                                     att.assign(anomaly_type="ABSENCE_STREAK"))
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(da, db)


def test_item_id_prefix_cannot_influence_reasons_or_amounts(planted_panel, planted_attendance):
    pay, _ = planted_panel
    att, _ = planted_attendance
    a, _, _ = build_official_output(pay, att)
    renamed = pay.assign(payroll_item_id=["PI-D" + p[1:] if i % 2 else p for i, p in enumerate(pay.payroll_item_id)])
    b, _, _ = build_official_output(renamed, att)
    # ids differ, everything the explanation is based on must not
    assert sorted(zip(a.reason_code, a.money_at_risk.fillna(-1))) == sorted(zip(b.reason_code, b.money_at_risk.fillna(-1)))


# ---------------------------------------------------------------- privacy
def test_detail_report_masks_employee_ids_and_official_file_has_no_extra_columns():
    official, detail, pay = small_official()
    assert "employee_id" not in detail.columns and detail["employee_ref"].str.startswith("EMP-").all()
    assert not set(detail["employee_ref"]) & set(pay["employee_id"])
    assert len(official.columns) == 12


def test_review_sheet_has_no_employee_ids_and_blank_human_columns(planted_panel, planted_attendance):
    official, _, _ = build_official_output(planted_panel[0], planted_attendance[0])
    sample = official[official["result_id"].isin(select_sample(official, 20))]
    sheet = build_review_sheet(sample)
    assert "employee_id" not in sheet.columns
    assert not any(str(v).startswith("EMP-") for v in sheet.astype(str).to_numpy().ravel())
    for col in HUMAN_COLUMNS:
        assert (sheet[col] == "").all(), "human feedback columns must be left empty; feedback may not be invented"
    assert "automated_lint (not human feedback)" in sheet.columns


# ---------------------------------------------------------------- the 20-row sample
def test_sample_is_real_deterministic_and_never_padded(planted_panel, planted_attendance):
    official, _, _ = build_official_output(planted_panel[0], planted_attendance[0])
    ids = select_sample(official, 20)
    assert len(ids) == len(set(ids)) == min(20, len(official)) and set(ids) <= set(official["result_id"])
    for seed in (3, 4, 5):                                                              # the order of the rows must not matter
        assert ids == select_sample(official.sample(frac=1, random_state=seed), 20)
    small = official.iloc[:5]
    assert select_sample(small, 20) == list(small["result_id"])                          # fewer than n -> all of them, no padding


def test_sample_covers_every_reason_code_present_when_there_is_room(planted_panel, planted_attendance):
    official, _, _ = build_official_output(planted_panel[0], planted_attendance[0])
    ids = select_sample(official, 20)
    present = set(official["reason_code"])
    assert len(present) <= 20
    assert present <= set(official.loc[official["result_id"].isin(ids), "reason_code"])


def test_lint_flags_jargon_and_missing_next_step():
    assert lint_text("Please check this payslip.") == "PASS"
    assert lint_text("The Isolation Forest score is unusual. Please check.").startswith("FAIL")
    assert lint_text("This payslip is odd.").startswith("FAIL")
