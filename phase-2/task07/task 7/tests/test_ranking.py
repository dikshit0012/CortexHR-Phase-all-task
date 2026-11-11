"""Ranking by money at risk: descending, empty amounts last, deterministic tie-breaks, independent of row order."""
import numpy as np
import pandas as pd
import pytest

from conftest import payslip, stub_flags
from cortexhr_task6.detection import run_detection
from cortexhr_task6.payroll_features import build_payroll_features
from cortexhr_task7.explain import explain_payroll
from cortexhr_task7.pipeline import assemble
from cortexhr_task7.ranking import rank_flags


def rec(rid, money, code="ZERO_PAY", signals=1, score=0.5, source="payroll_monthly_features"):
    return dict(source_dataset=source, record_id=rid, employee_id="E1", period=pd.Timestamp("2025-06-01"),
                primary_reason_code=code, reason_text="Please check.", money_at_risk=money, signal_count=signals,
                anomaly_score_raw=score, anomaly_score=round(score, 3))


def order(rows):
    return list(rank_flags(pd.DataFrame(rows))["record_id"])


def test_sorted_by_money_descending_with_empty_amounts_last():
    assert order([rec("a", 10.0), rec("b", np.nan, "STATISTICAL_OUTLIER"), rec("c", 500.0), rec("d", 0.0)]) == ["c", "a", "d", "b"]


def test_equal_money_is_broken_by_reason_priority_first():
    # NEGATIVE_NET_PAY is more serious than SALARY_DROP in the documented priority
    assert order([rec("x", 100.0, "SALARY_DROP"), rec("y", 100.0, "NEGATIVE_NET_PAY")]) == ["y", "x"]


def test_then_by_number_of_independent_signals():
    assert order([rec("x", 100.0, signals=1), rec("y", 100.0, signals=3)]) == ["y", "x"]


def test_then_by_detector_score():
    assert order([rec("x", 100.0, score=0.2), rec("y", 100.0, score=0.9)]) == ["y", "x"]


def test_final_tie_break_is_the_record_id_not_the_row_position():
    rows = [rec("b", 100.0), rec("a", 100.0), rec("c", 100.0)]
    assert order(rows) == order(rows[::-1]) == ["a", "b", "c"]


def test_unknown_reason_code_cannot_be_ranked():
    with pytest.raises(ValueError, match="unknown reason code"):
        rank_flags(pd.DataFrame([rec("a", 1.0, "MADE_UP")]))


def test_ranking_handles_no_flags():
    assert rank_flags(pd.DataFrame([rec("a", 1.0)]).iloc[0:0]).empty


def test_amount_is_not_confused_with_severity_or_confidence():
    # a huge amount from a low-priority, single-signal flag still ranks first: amount decides, the others only break ties
    rows = [rec("big", 1_000_000.0, "SALARY_DROP", signals=1), rec("small", 10.0, "DUPLICATE_PAYMENT", signals=3)]
    assert order(rows) == ["big", "small"]


def test_empty_flag_set_gives_an_empty_official_output_with_the_exact_columns():
    feats = build_payroll_features(pd.DataFrame([payslip("P1"), payslip("P2", emp="E2")]))
    flags = stub_flags(feats)
    pay = explain_payroll(feats, flags)
    assert pay.empty
    official, detail = assemble(pay, pay.copy())
    assert official.empty and len(official.columns) == 12 and detail.empty


# ---------------------------------------------------------------- stability under shuffling
def _snapshot(official):
    return list(zip(official["result_id"], official["record_id"], official["reason_code"], official["money_at_risk"]))


def test_assemble_is_identical_when_the_input_rows_are_shuffled(planted_panel):
    df, _ = planted_panel
    results = []
    for seed in (None, 1, 2):
        d = df if seed is None else df.sample(frac=1, random_state=seed).reset_index(drop=True)
        feats = build_payroll_features(d)
        flags = stub_flags(feats, model_flags=(feats["payroll_item_id"].str.endswith("7")).astype(int).to_numpy())
        pay = explain_payroll(feats, flags)
        official, detail = assemble(pay, pay.iloc[0:0])
        results.append(official)
    pd.testing.assert_frame_equal(results[0], results[1])
    pd.testing.assert_frame_equal(results[0], results[2])
    assert len(results[0]) > 5


def test_full_task6_pipeline_plus_explanation_is_stable_when_rows_are_shuffled(planted_panel):
    df, _ = planted_panel
    maps = []
    for seed in (None, 7):
        d = df if seed is None else df.sample(frac=1, random_state=seed).reset_index(drop=True)
        feats, flags, _ = run_detection(d)
        out = explain_payroll(feats, flags)
        maps.append({r.record_id: (r.primary_reason_code, r.money_at_risk, r.anomaly_score) for r in out.itertuples()})
    assert maps[0].keys() == maps[1].keys()
    for k in maps[0]:
        a, b = maps[0][k], maps[1][k]
        assert a[0] == b[0] and (a[1] == b[1] or (pd.isna(a[1]) and pd.isna(b[1]))) and a[2] == b[2]
