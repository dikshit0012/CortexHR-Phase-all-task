"""Task 6 - independent reconciliation of the saved reports.

Recomputes precision / recall / F1 for the three approaches from the labelled flagged-records file using
plain pandas (no cortexhr_task6 code), and compares them with task6_metrics_per_type.csv.
Exit code 0 = everything reconciles, 1 = at least one mismatch.

Usage:
    python "task 6/scripts/verify_task6_reports.py" [reports_dir]
"""
import json
import sys
from pathlib import Path

import pandas as pd

APPROACH_COLUMN = {"Rules only": "rules_only_flag", "Isolation Forest only": "if_flag", "Combined (rules OR model)": "combined_flag"}


def verify(reports_dir) -> list:
    d = Path(reports_dir)
    flagged = pd.read_csv(d / "task6_flagged_records_with_labels.csv")
    metrics = pd.read_csv(d / "task6_metrics_per_type.csv")
    baseline = pd.read_csv(d / "task6_baseline_comparison_overall.csv")
    sensitivity = pd.read_csv(d / "task6_sensitivity.csv")
    variants = pd.read_csv(d / "task6_original_vs_current_vs_exploratory.csv")
    salary_review = pd.read_csv(d / "task6_salary_drop_review.csv")
    summary = json.loads((d / "task6_run_summary.json").read_text(encoding="utf-8"))
    n_labels = summary["join_validation"]["label_rows"]
    n_rows = summary["join_validation"]["feature_rows"]
    n_clean = n_rows - n_labels
    problems = []
    for filename, table, fp_col, tn_col in [
        ("task6_metrics_per_type.csv", metrics, "fp", "tn"),
        ("task6_baseline_comparison_overall.csv", baseline, "fp", "tn"),
        ("task6_sensitivity.csv", sensitivity, "fp", "tn"),
        ("task6_original_vs_current_vs_exploratory.csv", variants, "fp", "tn"),
        ("task6_salary_drop_review.csv", salary_review, "false_positives", "true_negatives"),
    ]:
        if fp_col not in table or tn_col not in table:
            problems.append(f"{filename}: missing false-positive or true-negative column")
        elif not (table[tn_col] == n_clean - table[fp_col]).all():
            problems.append(f"{filename}: true negatives do not equal unlabelled unflagged records")

    # the combined flag must be exactly "any rule OR model" in the saved evidence
    if ((flagged["rules_only_flag"] | flagged["if_flag"]) != flagged["combined_flag"]).any():
        problems.append("combined_flag != rules_only_flag OR if_flag")
    rule_cols = [c for c in flagged.columns if c.startswith("rule_") and c != "rule_count"]
    if ((flagged[rule_cols].sum(axis=1) > 0).astype(int) != flagged["rules_only_flag"]).any():
        problems.append("rules_only_flag does not match the individual rule flags")

    for approach, col in APPROACH_COLUMN.items():
        f = flagged[flagged[col] == 1]
        tp = int((f["label_error_type"] != "NO_LABEL").sum())
        fp = len(f) - tp
        fn = n_labels - tp
        tn = n_clean - fp
        p, r = tp / len(f), tp / n_labels
        f1 = 2 * p * r / (p + r)
        row = metrics[(metrics.approach == approach) & (metrics.error_type == "ALL_ERRORS")].iloc[0]
        for name, mine, theirs in [("flagged", len(f), row.predicted_positive), ("tp", tp, row.tp), ("fp", fp, row.fp),
                                   ("fn", fn, row.fn), ("tn", tn, row.tn), ("precision", p, row.precision),
                                   ("recall", r, row.recall), ("f1", f1, row.f1)]:
            if abs(mine - theirs) > 1e-9:
                problems.append(f"{approach} ALL_ERRORS {name}: recomputed {mine} vs report {theirs}")
        # per type: true positives counted from the flagged file
        per_type_tp = f[f["label_error_type"] != "NO_LABEL"].groupby("label_error_type").size()
        sub = metrics[(metrics.approach == approach) & (metrics.error_type != "ALL_ERRORS")]
        for _, t in sub.iterrows():
            mine = int(per_type_tp.get(t.error_type, 0))
            if mine != t.tp or t.tp + t.fn != t.actual_cases:
                problems.append(f"{approach} {t.error_type}: tp {mine} vs {t.tp}, tp+fn {t.tp + t.fn} vs actual {t.actual_cases}")
            if t.tn != n_clean - t.fp:
                problems.append(f"{approach} {t.error_type}: tn {t.tn} vs clean-unflagged {n_clean - t.fp}")
        if int(sub.actual_cases.sum()) != n_labels:
            problems.append(f"{approach}: per-type actual cases do not sum to {n_labels}")
    return problems


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "reports"
    issues = verify(target)
    if issues:
        print("MISMATCHES FOUND:")
        for i in issues:
            print(" -", i)
        sys.exit(1)
    print(f"OK: reports in {target} reconcile with an independent recomputation (3 approaches x ALL_ERRORS and per type).")
