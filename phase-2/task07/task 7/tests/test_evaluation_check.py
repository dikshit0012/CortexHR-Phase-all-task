"""Evaluation-only checks: correct on small hand-made cases and safe when nothing matches."""
import pandas as pd

from cortexhr_task7.evaluation_check import money_vs_reference, reason_vs_label, reference_amount

PAY = "payroll_monthly_features"


def official(rows):
    return pd.DataFrame(rows, columns=["source_dataset", "record_id", "employee_id", "period", "reason_code", "money_at_risk"])


def test_reason_vs_label_counts_each_pair_and_marks_unlabelled_flags():
    off = official([(PAY, "P1", "E1", "2025-06-01", "ZERO_PAY", 10.0), (PAY, "P2", "E2", "2025-06-01", "ZERO_PAY", 5.0),
                    (PAY, "P3", "E3", "2025-06-01", "SALARY_DROP", 1.0), ("attendance_monthly_features", "A1", "E4", "2025-06-01", "PATTERN_SHIFT", None)])
    labels = pd.DataFrame({"payroll_item_id": ["P1"], "anomaly_type": ["ZERO_PAY"]})
    t = reason_vs_label(off, labels).set_index("reason_code")
    assert t.loc["ZERO_PAY", "ZERO_PAY"] == 1 and t.loc["ZERO_PAY", "NO_LABEL"] == 1 and t.loc["SALARY_DROP", "NO_LABEL"] == 1
    assert "PATTERN_SHIFT" not in t.index                                      # attendance flags are not part of the payroll label check


def test_money_vs_reference_empty_when_no_labelled_record_was_flagged():
    off = official([(PAY, "P9", "E9", "2025-06-01", "SALARY_DROP", 1.0)])
    raw = pd.DataFrame({"payroll_item_id": ["P1"], "employee_id": ["E1"], "month": [pd.Timestamp("2025-06-01")], "payable_days": [30.0],
                        "net_pay": [1.0], "tds": [1.0], "overtime_pay": [0.0], "monthly_fixed_gross": [1.0], "gross_pay": [1.0]})
    labels = pd.DataFrame({"payroll_item_id": ["P1"], "anomaly_type": ["ZERO_PAY"], "original_value": ["1.0"]})
    assert money_vs_reference(off, raw, labels).empty


def test_reference_amounts_follow_the_label_description():
    row = pd.Series({"month": pd.Timestamp("2025-04-01"), "payable_days": 15.0, "net_pay": 900.0, "tds": 700.0, "overtime_pay": 400.0,
                     "monthly_fixed_gross": 90000.0, "gross_pay": 1100.0})
    assert reference_amount("DOUBLE_PAYMENT", row, "single payment") == 900.0
    assert reference_amount("MISSED_DEDUCTION", row, "1800.00") == 1800.0
    assert reference_amount("NEGATIVE_NET", row, "0.00") == 700.0
    assert reference_amount("OVERTIME_SPIKE", row, "100.00") == 300.0
    assert reference_amount("ROLE_MISMATCH", row, "40000.00") == 25000.0           # (90,000 - 40,000) x 15/30
    assert reference_amount("STALE_SALARY", row, "1250.00") == 150.0
    assert reference_amount("WRONG_DECIMAL", row, "165489.00 / 147700.62") == 1100.0 - 165489.0
