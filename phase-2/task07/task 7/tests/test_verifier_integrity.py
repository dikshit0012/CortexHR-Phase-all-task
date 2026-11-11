"""Standalone checks for the report verifier; fixtures use no Task 5/Task 6 code."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexhr_task7.reason_codes import PRIORITY, REASON_CODES

VERIFY = Path(__file__).resolve().parents[1] / "scripts" / "verify_task7_reports.py"
SPEC = importlib.util.spec_from_file_location("verify_task7_reports_standalone", VERIFY)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def _reports(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    workbook = tmp_path / "workbook.xlsx"

    payroll = pd.DataFrame({
        "payroll_item_id": ["P-GROSS", "P-INVALID", "P-LOW", "P-HIGH", "P-SIGNALS", "P-A"],
        "employee_id": ["E0", "E1", "E2", "E3", "E4", "E5"],
        "month": pd.to_datetime(["2025-01-01"] * 6),
        "monthly_fixed_gross": [100.0] * 6,
        "payable_days": [31.0] * 6,
        "overtime_pay": [0.0] * 6,
        "variable_pay": [0.0] * 6,
        "total_deductions": [10.0] * 6,
        "gross_pay": [150.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "pf_employee": [0.0] * 6,
        "role_id": ["R1"] * 6,
        "net_pay": [140.0, 90.0, 90.0, 90.0, 90.0, 90.0],
        "tds": [10.0] * 6,
    })
    attendance = pd.DataFrame({
        "attendance_feature_id": ["A-1"],
        "employee_id": ["E6"],
        "month": pd.to_datetime(["2025-01-01"]),
    })
    official = pd.DataFrame([
        ["AN-0001", verifier.PAY, "P-GROSS", "E0", "2025-01-01", 0.2, "Y",
         "GROSS_ABOVE_NORM", "Please check.", 50.0, "version", "2026-10-01 02:00:00"],
        ["AN-0002", verifier.PAY, "P-INVALID", "E1", "2025-01-01", 0.1, "Y",
         "INVALID_PAYROLL_DATA", "Please check.", None, "version", "2026-10-01 02:00:00"],
        ["AN-0003", verifier.PAY, "P-SIGNALS", "E4", "2025-01-01", 0.1, "Y",
         "STATISTICAL_OUTLIER", "Please check.", None, "version", "2026-10-01 02:00:00"],
        ["AN-0004", verifier.PAY, "P-HIGH", "E3", "2025-01-01", 0.9, "Y",
         "STATISTICAL_OUTLIER", "Please check.", None, "version", "2026-10-01 02:00:00"],
        ["AN-0005", verifier.PAY, "P-A", "E5", "2025-01-01", 0.1, "Y",
         "STATISTICAL_OUTLIER", "Please check.", None, "version", "2026-10-01 02:00:00"],
        ["AN-0006", verifier.PAY, "P-LOW", "E2", "2025-01-01", 0.1, "Y",
         "STATISTICAL_OUTLIER", "Please check.", None, "version", "2026-10-01 02:00:00"],
        ["AN-0007", verifier.ATT, "A-1", "E6", "2025-01-01", 0.1, "Y",
         "PATTERN_SHIFT", "Please check.", None, "version", "2026-10-01 02:00:00"],
    ], columns=verifier.COLUMNS)
    sample = official.iloc[[0, 1, 2]].copy()
    dictionary = pd.DataFrame([
        [code, index + 1, verifier.REASON_SOURCE[code]]
        for index, code in enumerate(verifier.REASON_PRIORITY)
    ], columns=["reason_code", "priority", "source"])
    details = pd.DataFrame([
        ["AN-0001", verifier.PAY, "P-GROSS", "GROSS_ABOVE_NORM", 1, 1, 0, 0.2, 0.2],
        ["AN-0002", verifier.PAY, "P-INVALID", "INVALID_PAYROLL_DATA", 1, 1, 0, 0.1, 0.1],
        ["AN-0003", verifier.PAY, "P-SIGNALS", "STATISTICAL_OUTLIER", 2, 1, 1, 0.1, 0.1],
        ["AN-0004", verifier.PAY, "P-HIGH", "STATISTICAL_OUTLIER", 1, 0, 1, 0.9, 0.9],
        ["AN-0005", verifier.PAY, "P-A", "STATISTICAL_OUTLIER", 1, 0, 1, 0.1, 0.1],
        ["AN-0006", verifier.PAY, "P-LOW", "STATISTICAL_OUTLIER", 1, 0, 1, 0.1, 0.1],
        ["AN-0007", verifier.ATT, "A-1", "PATTERN_SHIFT", 1, 0, 1, 0.1, 0.1],
    ], columns=[
        "result_id", "source_dataset", "record_id", "primary_reason_code", "signal_count",
        "rule_count", "model_flag", "anomaly_score_raw", "anomaly_score",
    ])
    with pd.ExcelWriter(workbook) as writer:
        payroll.to_excel(writer, sheet_name=verifier.PAY, index=False)
        attendance.to_excel(writer, sheet_name=verifier.ATT, index=False)
        pd.DataFrame(columns=verifier.COLUMNS).to_excel(writer, sheet_name="sample_anomaly_results", index=False)
    official.to_csv(reports / "task7_anomaly_results.csv", index=False)
    sample.to_csv(reports / "task7_sample_20_flags.csv", index=False)
    details.to_csv(reports / "task7_flag_details.csv", index=False)
    dictionary.to_csv(reports / "task7_reason_code_dictionary.csv", index=False)
    return reports, workbook


def test_verifier_accepts_correct_order_and_exact_reason_coverage_sample(tmp_path):
    reports, workbook = _reports(tmp_path)
    assert verifier.verify(reports, workbook, sample_size=3) == []


def test_verifier_vocabulary_stays_aligned_with_the_reason_code_module():
    assert verifier.REASON_PRIORITY == PRIORITY
    assert verifier.REASON_SOURCE == {code: reason.source for code, reason in REASON_CODES.items()}


def test_verifier_rejects_wrong_deterministic_tie_break_order(tmp_path):
    reports, workbook = _reports(tmp_path)
    detail_path = reports / "task7_flag_details.csv"
    details = pd.read_csv(detail_path)
    details.loc[details["result_id"] == "AN-0004", "anomaly_score_raw"] = 0.05
    details.to_csv(detail_path, index=False)
    assert any(
        "complete documented ranking" in issue
        for issue in verifier.verify(reports, workbook, sample_size=3)
    )


def test_verifier_rejects_a_real_but_incomplete_sample(tmp_path):
    reports, workbook = _reports(tmp_path)
    official = pd.read_csv(reports / "task7_anomaly_results.csv")
    official.iloc[[0, 1]].to_csv(reports / "task7_sample_20_flags.csv", index=False)
    assert any(
        "documented 3-row selection rule" in issue
        for issue in verifier.verify(reports, workbook, sample_size=3)
    )


def test_verifier_rejects_a_tampered_amount(tmp_path):
    reports, workbook = _reports(tmp_path)
    official_path = reports / "task7_anomaly_results.csv"
    official = pd.read_csv(official_path)
    official.loc[0, "money_at_risk"] += 1
    official.to_csv(official_path, index=False)
    assert any(
        "independent recomputation" in issue
        for issue in verifier.verify(reports, workbook, sample_size=3)
    )


def test_verifier_rejects_a_tampered_reason_priority_dictionary(tmp_path):
    reports, workbook = _reports(tmp_path)
    dictionary_path = reports / "task7_reason_code_dictionary.csv"
    dictionary = pd.read_csv(dictionary_path)
    dictionary.loc[dictionary["reason_code"] == "GROSS_ABOVE_NORM", "priority"] = 99
    dictionary.to_csv(dictionary_path, index=False)
    assert any(
        "priorities do not match" in issue
        for issue in verifier.verify(reports, workbook, sample_size=3)
    )
