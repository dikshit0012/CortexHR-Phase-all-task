"""Run Task 11: rank resumes against job descriptions, evaluate against human labels, write reports/.

    python scripts/run_task11.py [--data-dir DIR] [--reports-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cortexhr_task11 import paths  # noqa: E402
from cortexhr_task11.pipeline import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--reports-dir", type=Path, default=None)
    a = ap.parse_args()
    reports_dir = a.reports_dir or paths.REPORTS_DIR
    res = run(a.data_dir, reports_dir)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    s = res["summary"]
    print("== Counts ==");        print(s["counts"])
    print("\n== Ranking quality vs human labels (pool = labelled resumes per job) ==")
    print(res["quality"].round(3).to_string(index=False))
    print("\n== Ablation (report only; weights are fixed 60/40, not tuned) ==")
    print(res["ablation"].round(3).to_string(index=False))
    print("\n== Misrankings ==");   print(s["misrankings"])
    print("\n== Ties ==");          print(s["ties"])
    print("\n== Experience dependence ==");  print(s["experience_dependence"])
    print("\nReports written to", reports_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
