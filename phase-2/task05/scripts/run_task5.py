"""Task 5 - Complete end-to-end attendance anomaly detection.

This script:
  1. Loads attendance_monthly_features (NO labels)
  2. Trains IsolationForest with ~3% contamination
  3. Applies business rules (zero attendance, impossible hours)
  4. Combines model + rules
  5. THEN loads hidden labels to evaluate with catch_rate()
  6. Compares against anomaly_baselines
  7. Analyses missed anomaly types
  8. Produces a priority-ranked review list

Usage:
    python scripts/run_task5.py
"""
import argparse
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortexhr_task5.attendance_anomaly_detector import (
    DetectorConfig, ID_COL, EMPLOYEE_COL, MONTH_COL,
    MODEL_FEATURES, RULE_COLUMNS,
    run_detector, build_feature_frame, apply_rules,
)
from cortexhr_task5.evaluation import (
    load_hidden_labels, catch_rate, summarise, find_workbook,
)


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
SEED = 42
CONTAMINATION = 0.03


def baseline_replica(raw: pd.DataFrame, feature_names, contamination=0.03, seed=42):
    """Re-run the published baseline recipe (raw columns, 3%) to verify metric definitions."""
    X = raw[feature_names].apply(pd.to_numeric, errors="coerce")
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    pred = IsolationForest(contamination=contamination, random_state=seed).fit_predict(Xi)
    return set(raw.loc[pred == -1, ID_COL])


def main(workbook: Path | None = None):
    xlsx_path = find_workbook(workbook)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK 5 - Attendance Anomaly Detection (Unsupervised)")
    print("=" * 70)

    # ================================================================
    # STEP 1: Load data (NO labels)
    # ================================================================
    print("\n[LOAD] Loading attendance_monthly_features...")
    print(f"   Workbook: {xlsx_path}")
    raw = pd.read_excel(xlsx_path, sheet_name="attendance_monthly_features")
    print(f"   Rows: {len(raw)}, Columns: {list(raw.columns)}")
    print(f"   Employees: {raw[EMPLOYEE_COL].nunique()}, Months span: {raw[MONTH_COL].nunique()}")

    # ================================================================
    # STEP 2: Train Isolation Forest (label-free, ~3% contamination)
    # ================================================================
    print(f"\n[MODEL] Training IsolationForest (contamination={CONTAMINATION}, seed={SEED})...")
    cfg = DetectorConfig(contamination=CONTAMINATION, random_state=SEED)
    feats, flags, ranked, det = run_detector(raw, cfg)
    n = len(feats)

    print(f"   Features kept: {det.kept_features_}")
    print(f"   Features dropped (constant): {det.dropped_constant_}")
    print(f"   IF flagged: {flags['if_flag'].sum()} rows ({100*flags['if_flag'].mean():.1f}%)")
    print(f"   Rule flagged: {flags['rules_only_flag'].sum()} rows")
    print(f"   Combined flagged: {flags['combined_anomaly'].sum()} rows ({100*flags['combined_anomaly'].mean():.1f}%)")

    # ================================================================
    # STEP 3: Save predictions (BEFORE seeing labels)
    # ================================================================
    cols = [ID_COL, EMPLOYEE_COL, MONTH_COL]
    flagged_out = pd.concat([
        feats.loc[ranked.index, cols],
        ranked[["priority", "priority_points", "if_anomaly_score", "if_flag"]
               + RULE_COLUMNS + ["anomaly_reasons"]]
    ], axis=1)
    flagged_out["month"] = flagged_out["month"].dt.strftime("%Y-%m")
    flagged_out.to_csv(REPORTS_DIR / "task5_flagged_records.csv", index=False)
    print(f"\n[SAVE] Flagged records saved to reports/task5_flagged_records.csv")

    # ================================================================
    # STEP 4: NOW load labels and evaluate with catch_rate()
    # ================================================================
    print("\n" + "=" * 70)
    print("EVALUATION (labels loaded only now - never used for training)")
    print("=" * 70)

    labels = load_hidden_labels(xlsx_path)
    injected = set(labels[ID_COL])
    print(f"\n[LABELS] Injected anomalies: {len(injected)} records")
    if "anomaly_type" in labels.columns:
        print(f"   Types: {labels['anomaly_type'].value_counts().to_dict()}")

    # Load anomaly_baselines for comparison
    base = pd.read_excel(xlsx_path, sheet_name="anomaly_baselines")
    b = base[base["task"].str.contains("Attendance", case=False, na=False)].iloc[0]
    base_feats = [c.strip() for c in b["features"].split(",")]

    # Build detector sets
    ids = feats[ID_COL]
    sets = {
        "Isolation Forest (our features)": set(ids[flags["if_flag"] == 1]),
        "Rule-based only": set(ids[flags["rules_only_flag"] == 1]),
        "Combined (IF OR rules)": set(ids[flags["combined_anomaly"] == 1]),
    }

    # Build metrics table
    rows = []
    # Published baseline
    p_base, r_base = float(b["precision"]), float(b["recall"])
    rows.append({
        "detector": "Published baseline (anomaly_baselines)",
        "precision": p_base, "recall": r_base,
        "f1": round(2 * p_base * r_base / (p_base + r_base), 3),
        "true_positives": round(r_base * b["injected"]),
        "false_positives": int(b["flagged"] - round(r_base * b["injected"])),
        "false_negatives": int(b["injected"] - round(r_base * b["injected"])),
        "records_flagged": int(b["flagged"]),
        "pct_flagged": round(100 * b["flagged"] / n, 2),
    })

    # Baseline recipe re-run
    rows.append(summarise(
        "Baseline recipe re-run (raw 10 features, 3%)",
        baseline_replica(raw, base_feats, 0.03, SEED), injected, n
    ))

    # Our detectors
    for name, s in sets.items():
        # Use catch_rate() as specified in the task
        cr = catch_rate(s, injected)
        row = summarise(name, s, injected, n)
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORTS_DIR / "task5_metrics.csv", index=False)

    print("\n[METRICS] RESULTS TABLE:")
    print("-" * 100)
    print(metrics.to_string(index=False))
    print("-" * 100)

    # Highlight the combined result vs baseline
    combined_row = metrics[metrics["detector"].str.contains("Combined")].iloc[0]
    print(f"\n[OK] Combined Detector: Precision={combined_row['precision']:.3f}, "
          f"Recall={combined_row['recall']:.3f}, F1={combined_row['f1']:.3f}")
    print(f"     vs Baseline:       Precision={p_base:.3f}, "
          f"Recall={r_base:.3f}, F1={round(2*p_base*r_base/(p_base+r_base), 3):.3f}")

    # ================================================================
    # STEP 5: Missed-anomaly analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("MISSED ANOMALY ANALYSIS")
    print("=" * 70)

    comb = sets["Combined (IF OR rules)"]
    lab = labels.merge(feats, on=ID_COL, suffixes=("", "_f"))
    lab["caught"] = lab[ID_COL].isin(comb)
    lab["caught_by_if"] = lab[ID_COL].isin(sets["Isolation Forest (our features)"])
    lab["caught_by_rule"] = lab[ID_COL].isin(sets["Rule-based only"])

    show = ["attendance_rate", "unpaid_absence_days", "longest_unpaid_streak_days",
            "late_arrivals", "avg_hours_worked", "rate_change_vs_own_history"]

    rec = []
    for t, g in lab.groupby("anomaly_type"):
        miss, hit = g[~g["caught"]], g[g["caught"]]
        rec.append({
            "anomaly_type": t,
            "injected": len(g),
            "caught": len(hit),
            "missed": len(miss),
            "recall": round(len(hit) / len(g), 3),
            "caught_by_if": int(g["caught_by_if"].sum()),
            "caught_by_rules": int(g["caught_by_rule"].sum()),
            **{f"missed_mean_{c}": round(miss[c].mean(), 2) if len(miss) else np.nan for c in show},
            **{f"caught_mean_{c}": round(hit[c].mean(), 2) if len(hit) else np.nan for c in show},
        })

    missed_df = pd.DataFrame(rec)
    missed_df.to_csv(REPORTS_DIR / "task5_missed_anomalies.csv", index=False)
    lab.loc[~lab["caught"], [ID_COL, "anomaly_type"] + show].to_csv(
        REPORTS_DIR / "task5_missed_record_ids.csv", index=False
    )

    print("\n[ANALYSIS] Per-type breakdown:")
    print(missed_df[["anomaly_type", "injected", "caught", "missed", "recall",
                      "caught_by_if", "caught_by_rules"]].to_string(index=False))

    # ================================================================
    # STEP 6: Priority tier analysis (post-hoc, not used in training)
    # ================================================================
    print("\n" + "=" * 70)
    print("PRIORITY TIER ANALYSIS (post-hoc)")
    print("=" * 70)

    r2 = ranked.copy()
    r2["is_injected"] = feats.loc[r2.index, ID_COL].isin(injected)
    tier = r2.groupby("priority", observed=True).agg(
        flagged=("is_injected", "size"),
        true_anomalies=("is_injected", "sum")
    )
    tier["precision"] = (tier["true_anomalies"] / tier["flagged"]).round(3)
    tier.to_csv(REPORTS_DIR / "task5_priority_tiers.csv")
    print(tier)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("TASK 5 COMPLETE")
    print("=" * 70)
    print(f"  Reports saved in: {REPORTS_DIR}")
    print(f"  Files: task5_metrics.csv, task5_flagged_records.csv,")
    print(f"         task5_missed_anomalies.csv, task5_missed_record_ids.csv,")
    print(f"         task5_priority_tiers.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Task 5 attendance anomaly detector.")
    parser.add_argument("--workbook", type=Path, help="authorized CortexHR workbook (overrides CORTEXHR_WORKBOOK)")
    main(parser.parse_args().workbook)
