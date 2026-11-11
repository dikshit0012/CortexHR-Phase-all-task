"""Run Task 15 from the task15 folder: python scripts/run_task15.py [--workbook PATH] [--unlock PATH] [--reports-dir DIR].
The salary data and authorized unlock_CX-15 audit file are required for a passing validation run."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

from cortexhr_task15.pipeline import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", type=Path); ap.add_argument("--unlock", type=Path); ap.add_argument("--reports-dir", type=Path)
    a = ap.parse_args()
    r = run(a.workbook, a.reports_dir, a.unlock)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(r["comparison"].round(3).to_string(index=False)); print("selected:", r["selected"]); print("band coverage:", r["summary"]["band"]["test_coverage"])
    print("gender audit run:", r["summary"]["gender_audit"]["run"])
    for c in r["validation"]:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    return 0 if r["summary"]["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
