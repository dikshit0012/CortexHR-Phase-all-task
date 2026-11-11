"""Task 7 - Explain Every Flag: reason codes and money at risk.

Runs the EXISTING Task 6 (payroll) and Task 5 (attendance) detectors unchanged, explains every flag, ranks the
flags by money at risk, and writes the official-format output (sheet `sample_anomaly_results`).

Usage (from the repository root):
    python "task07/task 7/scripts/run_task7.py"
    python "task07/task 7/scripts/run_task7.py" --workbook path/to/02_CortexHR_AI_ML_Dataset.xlsx --scored-at "2026-10-01 02:00:00"

Labels are read only AFTER all outputs exist, and only for the evaluation-only checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cortexhr_task7  # noqa: F401  (puts Task 5 / Task 6 on the path)
from cortexhr_task6.data_loading import find_workbook
from cortexhr_task6.payroll_evaluation import load_payroll_labels
from cortexhr_task7.evaluation_check import money_vs_reference, reason_vs_label
from cortexhr_task7.official_output import (
    DEFAULT_SCORED_AT, OFFICIAL_COLUMNS, sample_columns, validate_official_schema, validate_official_values,
)
from cortexhr_task7.pipeline import build_official_output
from cortexhr_task7.reason_codes import reason_code_dictionary
from cortexhr_task7.review_pack import build_review_sheet, select_sample


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Task 7: reason codes and money at risk")
    p.add_argument("--workbook", default=None, help="path to 02_CortexHR_AI_ML_Dataset.xlsx (default: auto-detect)")
    p.add_argument("--reports-dir", default=str(Path(__file__).resolve().parents[1] / "reports"))
    p.add_argument("--scored-at", default=DEFAULT_SCORED_AT, help="timestamp written to scored_at (fixed by default so reruns are identical)")
    p.add_argument("--sample-size", type=int, default=20)
    p.add_argument("--skip-label-check", action="store_true", help="skip the evaluation-only label checks")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    print("=" * 72 + "\nTASK 7 - Explain Every Flag\n" + "=" * 72)

    wb = find_workbook(args.workbook)
    print(f"[LOAD] workbook: {wb}")
    payroll_raw = pd.read_excel(wb, sheet_name="payroll_monthly_features")
    attendance_raw = pd.read_excel(wb, sheet_name="attendance_monthly_features")
    print(f"[LOAD] payroll rows: {len(payroll_raw)} | attendance rows: {len(attendance_raw)} (no labels loaded)")

    official, detail, info = build_official_output(payroll_raw, attendance_raw, scored_at=args.scored_at)
    print(f"[FLAGS] explained {len(official)} flagged records: {info['expected_counts']}")

    # --- schema check against the real sample sheet in the workbook
    expected = sample_columns(wb)
    validate_official_schema(official, expected)
    assert expected == OFFICIAL_COLUMNS, "OFFICIAL_COLUMNS constant no longer matches the sample sheet"
    print(f"[SCHEMA] output columns match the sample sheet exactly, in order: {list(official.columns)}")

    problems = validate_official_values(official)
    assert not problems, problems
    assert (official["reason_code"].str.len() > 0).all() and (official["reason_text"].str.len() > 0).all()
    money = official["money_at_risk"]
    n_est = int(money.notna().sum())
    ranked_ok = bool(money.iloc[:n_est].notna().all() and money.iloc[:n_est].is_monotonic_decreasing and money.iloc[n_est:].isna().all())
    assert ranked_ok, "flags are not ranked by money_at_risk (largest first, empty amounts last)"
    print("[CHECK] every flag has a reason code and text; amounts are >= 0; ranking is largest-first with empty amounts last")

    # --- write outputs (label-free)
    official.to_csv(reports / "task7_anomaly_results.csv", index=False)
    detail.to_csv(reports / "task7_flag_details.csv", index=False)
    reason_code_dictionary().to_csv(reports / "task7_reason_code_dictionary.csv", index=False)
    ids = select_sample(official, args.sample_size)
    sample = official[official["result_id"].isin(ids)]
    sample.to_csv(reports / "task7_sample_20_flags.csv", index=False)
    build_review_sheet(sample).to_csv(reports / "task7_human_review_sheet.csv", index=False)
    print(f"[SAVE] official output ({len(official)} rows), detail report, dictionary, {len(sample)}-row sample, human review sheet")

    # --- evaluation-only checks (labels first read here)
    summary = {
        "workbook": wb.name, "scored_at": args.scored_at, "rows_official": int(len(official)),
        "rows_by_source": official["source_dataset"].value_counts().to_dict(),
        "rows_by_reason_code": official["reason_code"].value_counts().to_dict(),
        "rows_with_money_estimate": int(money.notna().sum()), "rows_without_money_estimate": int(money.isna().sum()),
        "sum_of_estimates_inr_exposure_to_verify_NOT_confirmed_loss": round(float(money.sum()), 2),
        "sample_rows": int(len(sample)), "human_review_status": "PENDING - no human feedback has been recorded",
        "schema_matches_sample_sheet": True, "labels_loaded_before_outputs_written": False,
        "labels_used_for_reasons_or_amounts": False,
    }
    if not args.skip_label_check:
        labels, label_sheet = load_payroll_labels(wb)
        rvl = reason_vs_label(official, labels)
        mvr = money_vs_reference(official, payroll_raw, labels)
        rvl.to_csv(reports / "task7_evalonly_reason_vs_label.csv", index=False)
        mvr.to_csv(reports / "task7_evalonly_money_vs_reference.csv", index=False)
        summary["evaluation_only_label_sheet"] = label_sheet
        summary["evaluation_only_note"] = "Labels were read after the outputs were written and never influence reasons, amounts or ranking."
        print(f"[EVAL-ONLY] label sheet '{label_sheet}': wrote reason-vs-label and money-vs-reference tables")
    (reports / "task7_run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nTASK 7 COMPLETE - outputs in", reports)


if __name__ == "__main__":
    main()
