"""End to end on a tiny SYNTHETIC workbook (no real payroll data): the documented command works and is reproducible."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from cortexhr_task7.official_output import OFFICIAL_COLUMNS

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_task7.py"
FILES = ["task7_anomaly_results.csv", "task7_flag_details.csv", "task7_reason_code_dictionary.csv", "task7_sample_20_flags.csv",
         "task7_human_review_sheet.csv", "task7_run_summary.json", "task7_evalonly_reason_vs_label.csv",
         "task7_evalonly_money_vs_reference.csv"]


def load_runner():
    spec = importlib.util.spec_from_file_location("run_task7", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_workbook(path, pay, att, sample_columns=None, labels=None):
    sample = pd.DataFrame([["AN-0001", "x", "x", "E1", "2025-06-01", 0.9, "Y", "X", "x", 1.0, "m", "2026-10-01 02:00:00"]],
                          columns=sample_columns or OFFICIAL_COLUMNS)
    if labels is None:
        labels = pd.DataFrame({"payroll_item_id": [pay.payroll_item_id.iloc[0]], "anomaly_type": ["ZERO_PAY"],
                               "field_changed": ["gross_pay, net_pay"], "original_value": ["50000.00"], "injection_note": ["x"]})
    with pd.ExcelWriter(path) as w:
        pay.to_excel(w, sheet_name="payroll_monthly_features", index=False)
        att.to_excel(w, sheet_name="attendance_monthly_features", index=False)
        sample.to_excel(w, sheet_name="sample_anomaly_results", index=False)
        labels.to_excel(w, sheet_name="payroll_anomaly_labels", index=False)
    return path


@pytest.fixture
def workbook(tmp_path, planted_panel, planted_attendance):
    return write_workbook(tmp_path / "wb.xlsx", planted_panel[0], planted_attendance[0])


def test_script_writes_all_outputs_in_the_official_format(workbook, tmp_path):
    out = tmp_path / "reports"
    load_runner().main(["--workbook", str(workbook), "--reports-dir", str(out)])
    for name in FILES:
        assert (out / name).exists(), name
    official = pd.read_csv(out / "task7_anomaly_results.csv")
    assert list(official.columns) == OFFICIAL_COLUMNS
    assert official["reason_code"].notna().all() and official["reason_text"].notna().all()
    summary = json.loads((out / "task7_run_summary.json").read_text(encoding="utf-8"))
    assert summary["human_review_status"].startswith("PENDING") and summary["labels_used_for_reasons_or_amounts"] is False
    assert summary["schema_matches_sample_sheet"] is True
    sample = pd.read_csv(out / "task7_sample_20_flags.csv")
    assert list(sample.columns) == OFFICIAL_COLUMNS and len(sample) == min(20, len(official)) and set(sample.record_id) <= set(official.record_id)


def test_script_output_is_reproducible(workbook, tmp_path):
    mod = load_runner()
    a, b = tmp_path / "a", tmp_path / "b"
    mod.main(["--workbook", str(workbook), "--reports-dir", str(a)])
    mod.main(["--workbook", str(workbook), "--reports-dir", str(b)])
    for name in FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_script_stops_if_the_sample_sheet_columns_differ_from_the_implementation(tmp_path, planted_panel, planted_attendance):
    reordered = OFFICIAL_COLUMNS[1:] + OFFICIAL_COLUMNS[:1]
    wb = write_workbook(tmp_path / "wb.xlsx", planted_panel[0], planted_attendance[0], sample_columns=reordered)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_runner().main(["--workbook", str(wb), "--reports-dir", str(tmp_path / "r")])


def test_scored_at_can_be_set_and_label_check_skipped(workbook, tmp_path):
    out = tmp_path / "r"
    load_runner().main(["--workbook", str(workbook), "--reports-dir", str(out), "--scored-at", "2027-01-02 03:04:05", "--skip-label-check"])
    official = pd.read_csv(out / "task7_anomaly_results.csv")
    assert (official["scored_at"] == "2027-01-02 03:04:05").all()
    assert not (out / "task7_evalonly_reason_vs_label.csv").exists()


def test_label_check_reads_labels_only_after_outputs_exist(workbook, tmp_path):
    out = tmp_path / "r"
    with_labels = json.loads(_run(workbook, out)["summary"])
    assert with_labels["labels_loaded_before_outputs_written"] is False
    # outputs are identical whether or not the label check runs: labels cannot influence them
    out2 = tmp_path / "r2"
    load_runner().main(["--workbook", str(workbook), "--reports-dir", str(out2), "--skip-label-check"])
    for name in ("task7_anomaly_results.csv", "task7_flag_details.csv", "task7_sample_20_flags.csv"):
        assert (out / name).read_bytes() == (out2 / name).read_bytes(), name


def _run(workbook, out):
    load_runner().main(["--workbook", str(workbook), "--reports-dir", str(out)])
    return {"summary": (out / "task7_run_summary.json").read_text(encoding="utf-8")}
