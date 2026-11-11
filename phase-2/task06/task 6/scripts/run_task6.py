"""Task 6 - Payroll Outlier Detector: end-to-end run.

Steps
  1. Load payroll_monthly_features (NO labels).
  2. Engineer ratio features, apply rules, fit Isolation Forest, combine.
  3. Save label-free flagged records (identifiers masked).
  4. ONLY NOW load the labels and evaluate rules / model / combined per error type.
  5. Compare with anomaly_baselines, analyse rules-vs-model, run post-hoc sensitivity.
  6. Write CSV / JSON evidence and the Markdown report.

Usage (from the repository root):
    python "task 6/scripts/run_task6.py"
    python "task 6/scripts/run_task6.py" --workbook path/to/02_CortexHR_AI_ML_Dataset.xlsx --contamination 0.025 --seed 42
    python "task 6/scripts/run_task6.py" --original-experiment --reports-dir "task 6/reports/original_experiment_rerun"

--original-experiment reproduces the FIRST experiment (file-order duplicate rule, file-order model fit) so the
original numbers can be re-checked. The default is the corrected, row-order-independent implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortexhr_task6.data_loading import find_workbook, load_baseline_row, load_payroll_features
from cortexhr_task6.detection import run_detection
from cortexhr_task6.experiments import baseline_replica_seed_table, original_experiment_configs, salary_drop_review, tiered_alert_table, variant_comparison
from cortexhr_task6.payroll_evaluation import (
    ALL_ERRORS, ID_COL, LABEL_TYPE_COL, baseline_replica_flag, clean_overlap, load_payroll_labels,
    overlap_table, per_type_metrics, results_table, rule_hit_table, validate_label_join,
)
from cortexhr_task6.payroll_features import EMPLOYEE_COL, FORBIDDEN_MODEL_COLUMNS, MODEL_FEATURES, MONTH_COL, RoleReference
from cortexhr_task6.payroll_model import ModelConfig, assert_no_forbidden_columns
from cortexhr_task6.payroll_rules import RULE_COLUMNS, RuleConfig, rule_definitions
from cortexhr_task6.privacy import mask_ids
from cortexhr_task6.report_writer import build_report
from cortexhr_task6.reporting import (
    build_data_dictionary, data_quality_summary, diagnostics, feature_definition_table, masked_examples,
)
from cortexhr_task6.sensitivity import run_sensitivity

APPROACHES = {"Rules": "Rules only", "Model": "Isolation Forest only", "Combined": "Combined (rules OR model)"}
FLAG_COLUMN = {"Rules only": "rules_only_flag", "Isolation Forest only": "if_flag", "Combined (rules OR model)": "combined_flag"}
RATIO_COLUMNS = list(MODEL_FEATURES) + ["expected_gross"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Task 6 payroll outlier detector")
    p.add_argument("--workbook", default=None, help="path to 02_CortexHR_AI_ML_Dataset.xlsx (default: auto-detect)")
    p.add_argument("--reports-dir", default=str(Path(__file__).resolve().parents[1] / "reports"))
    p.add_argument("--contamination", type=float, default=0.025)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--original-experiment", action="store_true", help="reproduce the first (order-dependent) experiment")
    p.add_argument("--label-sheet", default=None, help="labels sheet name (default: unlock_CX-06, else payroll_anomaly_labels)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    reports = Path(args.reports_dir)
    private = reports / "private"
    reports.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    print("=" * 72 + "\nTASK 6 - Payroll Outlier Detector\n" + "=" * 72)
    wb = find_workbook(args.workbook)
    print(f"[LOAD] workbook: {wb}")
    raw = load_payroll_features(wb)
    print(f"[LOAD] payroll_monthly_features: {raw.shape[0]} rows x {raw.shape[1]} columns (no labels)")

    # ------------------------------------------------------------------ label-free detection
    model_cfg = ModelConfig(contamination=args.contamination, random_state=args.seed)
    rule_cfg = RuleConfig()
    if args.original_experiment:
        rule_cfg, model_cfg = original_experiment_configs(rule_cfg, model_cfg)
        print("[MODE] ORIGINAL EXPERIMENT settings (order-dependent duplicate rule and model fit)")
    assert_no_forbidden_columns(model_cfg.features)
    role_reference = RoleReference().fit(raw) if args.original_experiment else None
    feats, flags, model = run_detection(raw, rule_cfg, model_cfg, role_reference=role_reference)
    labels_loaded_at_prediction_time = "labels" in locals()   # False: labels are first read below
    predictions_done_at = datetime.now(timezone.utc).isoformat()
    print(f"[MODEL] features: {model.kept_features_}; dropped constant: {model.dropped_constant_}")
    print(f"[FLAGS] rules={flags['rules_only_flag'].sum()}  model={flags['if_flag'].sum()}  combined={flags['combined_flag'].sum()}")

    out = pd.concat([feats[[ID_COL, EMPLOYEE_COL, MONTH_COL]], feats[RATIO_COLUMNS], flags], axis=1)
    out["record_ref"] = mask_ids(out[ID_COL], "REC")
    out["employee_ref"] = mask_ids(out[EMPLOYEE_COL], "EMP")
    out["month"] = out[MONTH_COL].dt.strftime("%Y-%m")
    review_cols = (["record_ref", "employee_ref", "month"] + RULE_COLUMNS + ["rule_count", "rules_only_flag", "if_flag",
                   "if_anomaly_score", "combined_flag"] + RATIO_COLUMNS + ["detection_reasons"])
    flagged = out[out["combined_flag"] == 1].sort_values(["rule_count", "if_anomaly_score"], ascending=False)
    flagged[review_cols].to_csv(reports / "task6_flagged_records.csv", index=False)
    flagged[["record_ref", ID_COL, "employee_ref", EMPLOYEE_COL]].to_csv(private / "task6_record_key.csv", index=False)  # gitignored
    print(f"[SAVE] label-free flagged records -> task6_flagged_records.csv ({len(flagged)} rows)")

    # ------------------------------------------------------------------ evaluation (labels first loaded here)
    print("\n" + "=" * 72 + "\nEVALUATION (labels loaded only now, never used in fitting)\n" + "=" * 72)
    labels, label_sheet = load_payroll_labels(wb, args.label_sheet)
    join = validate_label_join(feats[ID_COL], labels)
    official_used = label_sheet == "unlock_CX-06"
    if not official_used:
        msg = (f"WARNING: the required evaluation sheet 'unlock_CX-06' is NOT available. Using '{label_sheet}' as a substitute. "
               "All evaluation results are PROVISIONAL until the internship team confirms the substitute is acceptable.")
        print("\n" + "!" * 72 + "\n" + msg + "\n" + "!" * 72 + "\n", file=sys.stderr)
        print("[WARN] " + msg)
    print(f"[LABELS] sheet '{label_sheet}': {join}")
    print(f"[LABELS] types: {labels[LABEL_TYPE_COL].value_counts().to_dict()}")
    if labels[ID_COL].duplicated().any():
        raise ValueError("a payslip carries more than one label - types would not be mutually exclusive")

    ids = feats[ID_COL].values
    series = {name: pd.Series(flags[col].values, index=ids) for name, col in FLAG_COLUMN.items()}
    per_type = pd.concat([per_type_metrics(name, s, labels) for name, s in series.items()], ignore_index=True)
    per_type.to_csv(reports / "task6_metrics_per_type.csv", index=False)
    wide = results_table(per_type, {"Model": "Isolation Forest only", "Rules": "Rules only", "Combined": "Combined (rules OR model)"})
    wide.to_csv(reports / "task6_results_table.csv", index=False)
    print("\n[RESULTS] per error type (precision/recall):")
    print(wide.round(3).to_string(index=False))

    overlap = overlap_table(series["Rules only"], series["Isolation Forest only"], labels)
    cleans = clean_overlap(series["Rules only"], series["Isolation Forest only"], labels)
    overlap.to_csv(reports / "task6_rules_vs_model_overlap.csv", index=False)
    fl_by_id = flags.copy()
    fl_by_id.index = ids
    rule_hits = rule_hit_table(fl_by_id, labels, RULE_COLUMNS)
    rule_hits.to_csv(reports / "task6_rule_hits.csv", index=False)
    print("\n[OVERLAP]\n" + overlap.to_string(index=False))
    print(f"[CLEAN ROWS] {cleans}")

    # ------------------------------------------------------------------ baseline comparison
    base = load_baseline_row(wb)
    inj, flg = int(base["injected"]), int(base["flagged"])
    tp_pub = round(float(base["recall"]) * inj)
    p_pub, r_pub = float(base["precision"]), float(base["recall"])
    rows = [{"detector": "Published baseline (anomaly_baselines sheet)", "flagged": flg, "tp": tp_pub, "fp": flg - tp_pub,
             "fn": inj - tp_pub, "tn": len(raw) - inj - (flg - tp_pub),
             "precision": p_pub, "recall": r_pub, "f1": 2 * p_pub * r_pub / (p_pub + r_pub)}]
    replicas = {}
    for label, keep in [("Baseline recipe replica (dup_flag marks every copy)", False), ("Baseline recipe replica (dup_flag marks 2nd+ copies)", "first")]:
        s = baseline_replica_flag(raw, contamination=0.025, seed=args.seed, dup_keep=keep)
        replicas[label] = s
        r = per_type_metrics(label, s, labels).query("error_type == @ALL_ERRORS").iloc[0]
        rows.append({"detector": label, "flagged": int(s.sum()), "tp": int(r.tp), "fp": int(r.fp), "fn": int(r.fn), "tn": int(r.tn),
                     "precision": r.precision, "recall": r.recall, "f1": r.f1})
    for name, s in series.items():
        r = per_type_metrics(name, s, labels).query("error_type == @ALL_ERRORS").iloc[0]
        rows.append({"detector": f"This work: {name}", "flagged": int(s.sum()), "tp": int(r.tp), "fp": int(r.fp), "fn": int(r.fn), "tn": int(r.tn),
                     "precision": r.precision, "recall": r.recall, "f1": r.f1})
    baseline_overall = pd.DataFrame(rows)
    baseline_overall.to_csv(reports / "task6_baseline_comparison_overall.csv", index=False)

    rep_name = "Baseline recipe replica (dup_flag marks every copy)"   # flagged count matches the published 184
    rep = per_type_metrics("baseline_replica", replicas[rep_name], labels)
    cmp_rows = []
    for t in sorted(labels[LABEL_TYPE_COL].unique()):
        g = lambda df, a: float(df[(df["approach"] == a) & (df["error_type"] == t)]["recall"].iloc[0])
        b = float(rep[rep["error_type"] == t]["recall"].iloc[0])
        m, ru, c = g(per_type, "Isolation Forest only"), g(per_type, "Rules only"), g(per_type, "Combined (rules OR model)")
        cmp_rows.append({"error_type": t, "actual_cases": int(rep[rep["error_type"] == t]["actual_cases"].iloc[0]),
                         "baseline_replica_recall": b, "model_recall": m, "rules_recall": ru, "combined_recall": c,
                         "delta_model_vs_baseline": m - b, "delta_combined_vs_baseline": c - b})
    baseline_per_type = pd.DataFrame(cmp_rows)
    baseline_per_type.to_csv(reports / "task6_baseline_comparison_per_type.csv", index=False)
    pub_ok = (abs(baseline_overall.iloc[1]["flagged"] - flg) <= 1) and (baseline_overall.iloc[1]["tp"] == tp_pub)
    baseline_note = (
        f"Published row: flagged {flg}, TP {tp_pub} (derived as recall x injected), precision {p_pub}, recall {r_pub}. "
        + ("The replica reproduces these figures." if pub_ok else
           f"The replica flags {int(baseline_overall.iloc[1]['flagged'])} payslips with TP {int(baseline_overall.iloc[1]['tp'])}, i.e. it does **not** reproduce the published TP exactly "
           "(the sheet does not state the `dup_flag` definition, imputation, seed or number of trees). Per-type baseline figures are therefore approximate; "
           "compare the pooled published row for the headline claim.")
    )
    print("\n[BASELINE]\n" + baseline_overall.round(3).to_string(index=False))

    # ------------------------------------------------------------------ diagnostics, examples, sensitivity
    diag = diagnostics(feats, flags, labels)
    diag["missed_rows"].to_csv(reports / "task6_missed_errors.csv", index=False)
    examples = masked_examples(feats, flags, labels)
    examples.to_csv(reports / "task6_examples_masked.csv", index=False)
    sens = run_sensitivity(raw, labels, model_cfg, rule_cfg)
    sens.to_csv(reports / "task6_sensitivity.csv", index=False)
    variants = variant_comparison(raw, labels, rule_cfg, model_cfg)
    variants.to_csv(reports / "task6_original_vs_current_vs_exploratory.csv", index=False)
    drop_review = salary_drop_review(raw, labels, rule_cfg, model_cfg)
    drop_review.to_csv(reports / "task6_salary_drop_review.csv", index=False)
    tiers = tiered_alert_table(flags, ids, labels)
    tiers.to_csv(reports / "task6_experimental_tiered_alerts.csv", index=False)
    replica_seeds = baseline_replica_seed_table(raw, labels)
    replica_seeds.to_csv(reports / "task6_baseline_replica_seeds.csv", index=False)
    print("\n[VARIANTS]\n" + variants[variants.approach == "combined"].round(3).to_string(index=False))
    print("\n[SALARY-DROP REVIEW]\n" + drop_review.round(3).to_string(index=False))

    # evaluation-only flagged file (labels kept OUT of the label-free file above)
    lab = labels.set_index(ID_COL)[LABEL_TYPE_COL]
    ev = flagged[review_cols].copy()
    ev.insert(3, "label_error_type", lab.reindex(flagged[ID_COL].values).fillna("NO_LABEL").values)
    ev.insert(4, "outcome", ["FALSE_ALARM" if t == "NO_LABEL" else "TRUE_POSITIVE" for t in ev["label_error_type"]])
    ev.to_csv(reports / "task6_flagged_records_with_labels.csv", index=False)

    # ------------------------------------------------------------------ documentation tables
    try:
        wb_dict = pd.read_excel(wb, sheet_name="data_dictionary")
    except Exception:  # dictionary sheet is optional
        wb_dict = None
    dd = build_data_dictionary(raw, wb_dict)
    dd.to_csv(reports / "task6_data_dictionary.csv", index=False)
    fdefs, rdefs = feature_definition_table(), rule_definitions(rule_cfg)
    fdefs.to_csv(reports / "task6_feature_definitions.csv", index=False)
    rdefs.to_csv(reports / "task6_rule_definitions.csv", index=False)
    dq = data_quality_summary(raw, feats)

    # ------------------------------------------------------------------ leakage / consistency checks
    comb_ok = bool(((flags["rules_only_flag"] | flags["if_flag"]) == flags["combined_flag"]).all())
    checks = {
        "model_features": model.kept_features_,
        "forbidden_columns_in_model_features": [c for c in model.kept_features_ if c in FORBIDDEN_MODEL_COLUMNS],
        "label_columns_in_model_features": [c for c in model.kept_features_ if c in labels.columns],
        "labels_loaded_at_prediction_time": labels_loaded_at_prediction_time,
        "predictions_completed_at_utc": predictions_done_at,
        "combined_equals_rules_or_model": comb_ok,
        "join_no_duplicates_no_drops": join["duplicate_label_keys"] == 0 and join["unmatched_rows"] == 0,
        "labels_mutually_exclusive": not labels[ID_COL].duplicated().any(),
    }
    assert not checks["labels_loaded_at_prediction_time"]
    assert not checks["forbidden_columns_in_model_features"] and not checks["label_columns_in_model_features"]
    assert comb_ok
    summary = {"workbook": str(wb.name), "label_sheet": label_sheet,
               "evaluation_resource": {"required": "unlock_CX-06", "used": label_sheet, "official_resource_used": official_used,
                                       "status": "final" if official_used else "PROVISIONAL - substitute labels, not confirmed by the internship team"},
               "feature_config": {"role_reference": "full input window (legacy experiment)" if args.original_experiment else "previous calendar month"},
               "mode": "original_experiment" if args.original_experiment else "corrected_default", "join_validation": join, "clean_row_overlap": cleans,
               "data_quality": dq, "model_config": model_cfg.__dict__ | {"features": model.kept_features_},
               "rule_config": rule_cfg.__dict__, "checks": checks}
    (reports / "task6_run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    ctx = dict(per_type=per_type, overall=baseline_overall, overlap=overlap, clean_overlap=cleans, rule_hits=rule_hits,
               sensitivity=sens, diagnostics=diag, join=join, data_quality=dq, model_config=model_cfg, rule_config=rule_cfg,
               label_sheet=label_sheet, feature_defs=fdefs, rule_defs=rdefs, kept_features=model.kept_features_,
               dropped_features=model.dropped_constant_, results_table=wide, baseline_overall=baseline_overall,
               baseline_note=baseline_note, baseline_per_type=baseline_per_type, examples=examples,
               official_label_resource=official_used, variants=variants, drop_review=drop_review,
               replica_seeds=replica_seeds, tiers=tiers, original_mode=args.original_experiment)
    (reports / "task6_comparison_report.md").write_text(build_report(ctx), encoding="utf-8")

    print("\n" + "=" * 72 + "\nTASK 6 COMPLETE\n" + "=" * 72)
    print(f"Outputs in {reports}")
    print("Checks:", json.dumps({k: v for k, v in checks.items() if k != 'model_features'}, default=str))


if __name__ == "__main__":
    main()
