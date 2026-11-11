"""End-to-end smoke test on a tiny SYNTHETIC workbook (no real payroll data)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import payslip

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_task6.py"

EXPECTED_FILES = [
    "task6_flagged_records.csv", "task6_flagged_records_with_labels.csv", "task6_metrics_per_type.csv",
    "task6_results_table.csv", "task6_rules_vs_model_overlap.csv", "task6_rule_hits.csv",
    "task6_baseline_comparison_overall.csv", "task6_baseline_comparison_per_type.csv", "task6_missed_errors.csv",
    "task6_examples_masked.csv", "task6_sensitivity.csv", "task6_data_dictionary.csv",
    "task6_feature_definitions.csv", "task6_rule_definitions.csv", "task6_run_summary.json", "task6_comparison_report.md",
    "task6_original_vs_current_vs_exploratory.csv", "task6_salary_drop_review.csv", "task6_baseline_replica_seeds.csv",
]


@pytest.fixture
def workbook(tmp_path):
    rows, n = [], 0
    rng = np.random.default_rng(1)
    for e in range(30):
        for m in range(1, 7):
            n += 1
            rows.append(payslip(f"PI-{n:04d}", emp=f"EMP-{e:03d}", month=f"2025-{m:02d}-01", role=f"R{e % 3}",
                                fixed=float(rng.integers(30, 90) * 1000)))
    df = pd.DataFrame(rows)
    df.loc[10, ["gross_pay", "net_pay"]] = [df.loc[10, "gross_pay"] * 10, df.loc[10, "net_pay"] * 10]
    df.loc[40, "net_pay"] = -500.0
    dup = df.iloc[[70]].copy()
    dup["payroll_item_id"] = "PI-D0001"
    df = pd.concat([df, dup], ignore_index=True)
    labels = pd.DataFrame({"payroll_item_id": [df.loc[10, "payroll_item_id"], df.loc[40, "payroll_item_id"], "PI-D0001"],
                           "anomaly_type": ["WRONG_DECIMAL", "NEGATIVE_NET", "DOUBLE_PAYMENT"]})
    base = pd.DataFrame({"task": ["Attendance anomalies", "Payroll anomalies (Isolation Forest on ratios, 2.5% contamination)"],
                         "features": ["x", "gross_vs_fixed, ded_ratio, net_ratio, overtime_hours_paid, dup_flag, net_pay"],
                         "precision": [0.4, 0.2], "recall": [0.6, 0.4], "flagged": [10, 6], "injected": [5, 3]})
    path = tmp_path / "wb.xlsx"
    with pd.ExcelWriter(path) as w:
        df.to_excel(w, sheet_name="payroll_monthly_features", index=False)
        labels.to_excel(w, sheet_name="payroll_anomaly_labels", index=False)
        base.to_excel(w, sheet_name="anomaly_baselines", index=False)
    return path


def load_runner():
    spec = importlib.util.spec_from_file_location("run_task6", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_writes_all_outputs_with_valid_content(workbook, tmp_path, capsys):
    out = tmp_path / "reports"
    load_runner().main(["--workbook", str(workbook), "--reports-dir", str(out)])
    captured = capsys.readouterr()
    assert "unlock_CX-06" in captured.err and "PROVISIONAL" in captured.err          # fallback is announced loudly
    report = (out / "task6_comparison_report.md").read_text(encoding="utf-8")
    assert "PROVISIONAL RESULTS" in report and "unlock_CX-06" in report                # ...and in the report
    assert "**not available**" in report                                              # never claims the official sheet was used
    assert "Post-evaluation" in report or "POST-EVALUATION" in report
    for name in EXPECTED_FILES:
        assert (out / name).exists(), name
    assert (out / "private" / "task6_record_key.csv").exists()          # raw-id mapping kept in the private folder

    pt = pd.read_csv(out / "task6_metrics_per_type.csv")
    assert set(pt.approach) == {"Rules only", "Isolation Forest only", "Combined (rules OR model)"}
    assert {"DOUBLE_PAYMENT", "NEGATIVE_NET", "WRONG_DECIMAL", "ALL_ERRORS"} <= set(pt.error_type)
    for col in ["actual_cases", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "support"]:
        assert col in pt.columns
    tn_columns = {
        "task6_baseline_comparison_overall.csv": "tn",
        "task6_sensitivity.csv": "tn",
        "task6_original_vs_current_vs_exploratory.csv": "tn",
        "task6_salary_drop_review.csv": "true_negatives",
    }
    for name, column in tn_columns.items():
        assert column in pd.read_csv(out / name).columns

    summary = json.loads((out / "task6_run_summary.json").read_text(encoding="utf-8"))
    assert summary["checks"]["labels_loaded_at_prediction_time"] is False
    assert summary["checks"]["combined_equals_rules_or_model"] is True
    assert summary["checks"]["forbidden_columns_in_model_features"] == []
    assert summary["join_validation"]["unmatched_rows"] == 0
    assert summary["evaluation_resource"]["official_resource_used"] is False
    assert "PROVISIONAL" in summary["evaluation_resource"]["status"]
    assert summary["mode"] == "corrected_default"

    flagged = pd.read_csv(out / "task6_flagged_records.csv")
    assert not any(c in flagged.columns for c in ["payroll_item_id", "employee_id", "anomaly_type", "label_error_type"])
    assert flagged["record_ref"].str.startswith("REC-").all() and flagged["employee_ref"].str.startswith("EMP-").all()
    assert not flagged["record_ref"].str.contains("PI-").any()
    with_labels = pd.read_csv(out / "task6_flagged_records_with_labels.csv")
    assert "label_error_type" in with_labels.columns and "outcome" in with_labels.columns


def test_pipeline_is_reproducible(workbook, tmp_path):
    mod = load_runner()
    a, b = tmp_path / "a", tmp_path / "b"
    mod.main(["--workbook", str(workbook), "--reports-dir", str(a)])
    mod.main(["--workbook", str(workbook), "--reports-dir", str(b)])
    for name in ["task6_metrics_per_type.csv", "task6_flagged_records.csv", "task6_comparison_report.md"]:
        assert (a / name).read_text(encoding="utf-8") == (b / name).read_text(encoding="utf-8"), name


def test_saved_reports_reconcile_with_independent_recomputation(workbook, tmp_path):
    verify_path = Path(__file__).resolve().parents[1] / "scripts" / "verify_task6_reports.py"
    spec = importlib.util.spec_from_file_location("verify_task6_reports", verify_path)
    verify_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify_mod)
    out = tmp_path / "reports"
    load_runner().main(["--workbook", str(workbook), "--reports-dir", str(out)])
    assert verify_mod.verify(out) == []
    # tamper with a metric and make sure the reconciliation notices
    m = pd.read_csv(out / "task6_metrics_per_type.csv")
    m.loc[(m.approach == "Rules only") & (m.error_type == "ALL_ERRORS"), "tp"] += 1
    m.to_csv(out / "task6_metrics_per_type.csv", index=False)
    assert verify_mod.verify(out) != []


def test_original_experiment_flag_runs_and_is_labelled(workbook, tmp_path):
    out = tmp_path / "orig"
    load_runner().main(["--workbook", str(workbook), "--reports-dir", str(out), "--original-experiment"])
    summary = json.loads((out / "task6_run_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "original_experiment"
    assert summary["rule_config"]["duplicate_selection"] == "file_order"
    assert summary["model_config"]["canonical_row_order"] is False
    assert "ORIGINAL EXPERIMENT" in (out / "task6_comparison_report.md").read_text(encoding="utf-8")
