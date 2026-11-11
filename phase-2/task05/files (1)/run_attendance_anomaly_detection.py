"""Run Task 5 end to end.

    python scripts/run_attendance_anomaly_detection.py --xlsx path/to/02_CortexHR_AI_ML_Dataset.xlsx

Order matters: (1) predictions are produced WITHOUT labels and written out,
(2) only then are the hidden labels loaded for measurement and error analysis.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexhr_task5.attendance_anomaly_detector import (  # noqa: E402
    DetectorConfig, ID_COL, EMPLOYEE_COL, MONTH_COL, MODEL_FEATURES, RULE_COLUMNS, run_detector,
)
from cortexhr_task5.evaluation import load_hidden_labels, summarise  # noqa: E402


def baseline_replica(raw: pd.DataFrame, feature_names, contamination=0.03, seed=42):
    """Re-run the published baseline recipe (raw columns, 3%) to check our metric definitions."""
    X = raw[feature_names].apply(pd.to_numeric, errors="coerce")
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    pred = IsolationForest(contamination=contamination, random_state=seed).fit_predict(Xi)
    return set(raw.loc[pred == -1, ID_COL])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=os.environ.get("CORTEXHR_XLSX"), help="path to the AI/ML workbook")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "reports"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.xlsx or not Path(args.xlsx).exists():
        sys.exit("Provide --xlsx (or set CORTEXHR_XLSX) pointing to the dataset workbook.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. label-free detection ----------------
    raw = pd.read_excel(args.xlsx, sheet_name="attendance_monthly_features")
    cfg = DetectorConfig(contamination=0.03, random_state=args.seed)
    feats, flags, ranked, det = run_detector(raw, cfg)
    n = len(feats)
    print(f"rows={n}  features kept={det.kept_features_}  dropped constant={det.dropped_constant_}")

    # Flagged-records file: pseudonymous ids only, no labels, no extra personal data.
    cols = [ID_COL, EMPLOYEE_COL, MONTH_COL]
    flagged_out = pd.concat([feats.loc[ranked.index, cols],
                             ranked[["priority", "priority_points", "if_anomaly_score", "if_flag"] + RULE_COLUMNS + ["anomaly_reasons"]]], axis=1)
    flagged_out["month"] = flagged_out["month"].dt.strftime("%Y-%m")
    flagged_out.to_csv(out / "task5_flagged_records.csv", index=False)

    # ---------------- 2. evaluation (labels loaded only now) ----------------
    labels = load_hidden_labels(args.xlsx)
    injected = set(labels[ID_COL])
    base = pd.read_excel(args.xlsx, sheet_name="anomaly_baselines")
    b = base[base["task"].str.startswith("Attendance")].iloc[0]
    base_feats = [c.strip() for c in b["features"].split(",")]

    ids = feats[ID_COL]
    sets = {
        "Isolation Forest (our features)": set(ids[flags["if_flag"] == 1]),
        "Rule-based only": set(ids[flags["rules_only_flag"] == 1]),
        "Combined (IF OR rules)": set(ids[flags["combined_anomaly"] == 1]),
    }
    rows = []
    p, r = float(b["precision"]), float(b["recall"])
    rows.append({"detector": "Published baseline (anomaly_baselines)", "precision": p, "recall": r,
                 "f1": round(2 * p * r / (p + r), 3), "true_positives": round(r * b["injected"]),
                 "false_positives": int(b["flagged"] - round(r * b["injected"])),
                 "false_negatives": int(b["injected"] - round(r * b["injected"])),
                 "records_flagged": int(b["flagged"]), "pct_flagged": round(100 * b["flagged"] / n, 2)})
    rows.append(summarise("Baseline recipe re-run here (raw 10 features, 3%)",
                          baseline_replica(raw, base_feats, 0.03, args.seed), injected, n))
    for name, s in sets.items():
        rows.append(summarise(name, s, injected, n))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "task5_metrics.csv", index=False)
    print(metrics.to_string(index=False))

    # ---------------- 3. missed-anomaly analysis ----------------
    comb = sets["Combined (IF OR rules)"]
    lab = labels.merge(feats, on=ID_COL, suffixes=("", "_f"))
    lab["caught"] = lab[ID_COL].isin(comb)
    lab["caught_by_if"] = lab[ID_COL].isin(sets["Isolation Forest (our features)"])
    lab["caught_by_rule"] = lab[ID_COL].isin(sets["Rule-based only"])
    show = ["attendance_rate", "unpaid_absence_days", "longest_unpaid_streak_days", "late_arrivals",
            "avg_hours_worked", "rate_change_vs_own_history"]
    rec = []
    for t, g in lab.groupby("anomaly_type"):
        miss, hit = g[~g["caught"]], g[g["caught"]]
        rec.append({
            "anomaly_type": t, "injected": len(g), "caught": len(hit), "missed": len(miss),
            "recall": round(len(hit) / len(g), 3),
            "caught_by_if": int(g["caught_by_if"].sum()), "caught_by_rules": int(g["caught_by_rule"].sum()),
            **{f"missed_mean_{c}": round(miss[c].mean(), 2) if len(miss) else np.nan for c in show},
            **{f"caught_mean_{c}": round(hit[c].mean(), 2) if len(hit) else np.nan for c in show},
        })
    pd.DataFrame(rec).to_csv(out / "task5_missed_anomalies.csv", index=False)
    lab.loc[~lab["caught"], [ID_COL, "anomaly_type"] + show].to_csv(out / "task5_missed_record_ids.csv", index=False)
    print(pd.DataFrame(rec)[["anomaly_type", "injected", "caught", "missed", "recall", "caught_by_if", "caught_by_rules"]].to_string(index=False))

    # ---------------- 4. do the priority tiers help? (post-hoc check, not used to build them) ----------------
    r2 = ranked.copy()
    r2["is_injected"] = feats.loc[r2.index, ID_COL].isin(injected)
    tier = r2.groupby("priority", observed=True).agg(flagged=("is_injected", "size"), true_anomalies=("is_injected", "sum"))
    tier["precision"] = (tier["true_anomalies"] / tier["flagged"]).round(3)
    tier.to_csv(out / "task5_priority_tiers.csv")
    print(tier)


if __name__ == "__main__":
    main()
