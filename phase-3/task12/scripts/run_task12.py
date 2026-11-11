"""Run Task 12: counterfactual twin audit of the Task 11 screener, fair screener, comparison, reports.

    python scripts/run_task12.py [--reports-dir DIR] [--stress-step N]
Needs ../task11 (code + data) and the confidential unlock_CX-12 file in task12/private/ (optional: protected-value analyses are skipped without it).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cortexhr_task12.fairness_audit import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-dir", type=Path, default=None)
    ap.add_argument("--stress-step", type=int, default=6, help="synthetic stress test uses every Nth resume (default 6 -> 31 resumes)")
    a = ap.parse_args()
    res = run(a.reports_dir, stress_step=a.stress_step)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    s = res["summary"]
    print("pairs:", s["n_pairs"], "| tolerance:", s["tolerance"], "| protected attrs:", s["protected_attributes_available"])
    print("\n== Twin gap, primary job per pair ==")
    print(pd.DataFrame(s["primary_job_gap_summary"]).T.to_string())
    print("\n== Gap reduction (%) vs baselines (NaN = baseline already 0) ==", s["gap_reduction_pct_primary"])
    print("\n== Ranking quality ==");  print(res["quality"].round(4).to_string(index=False))
    print("\n== Bias source ablation ==");  print(res["ablation"][["variant", "mean_gap_primary_job", "max_gap_primary_job", "nonzero_pairs_primary_job"]].to_string(index=False))
    print("\n== Synthetic stress test ==");  print(res["stress"].round(6).to_string(index=False))
    print("\nReports written to", a.reports_dir or "task12/reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
