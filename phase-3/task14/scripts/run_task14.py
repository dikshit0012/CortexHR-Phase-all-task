"""Run Task 14: score the 430 active employees with attrition-v2 and write the explained risk report.

    python scripts/run_task14.py [--reports-dir DIR] [--recompute-shap] [--no-stability]
Needs Task 13 (CORTEXHR_TASK13_ROOT, default ../task 13), the Task 2-4 package (CORTEXHR_ML_SRC) and the workbook (CORTEXHR_WORKBOOK).
SHAP for 430 employees takes ~12 minutes on one CPU core the first time; values are cached for identical inputs.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from cortexhr_task14.pipeline import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-dir", type=Path, default=None)
    ap.add_argument("--workbook", type=Path, default=None)
    ap.add_argument("--recompute-shap", action="store_true")
    ap.add_argument("--no-stability", action="store_true")
    a = ap.parse_args()
    res = run(a.workbook, a.reports_dir, recompute_shap=a.recompute_shap, stability_check=not a.no_stability)
    s = res["summary"]
    pd.set_option("display.width", 220, "display.max_colwidth", 200)
    print("model:", s["model"]["model_version"], "|", s["model"]["estimator"], "x", s["model"]["ensemble_members"], "| employees:", s["employees"])
    print("risk bands:", s["risk_bands"]["counts"], "| helper cut-offs p90/p70:", round(s["risk_bands"]["high_cutoff_p90"], 4), round(s["risk_bands"]["medium_cutoff_p70"], 4))
    print("reasons:", s["reasons"])
    print("top features:", [(f["feature"], round(f["mean_abs_shap_percentage_points"], 2)) for f in s["top_features"][:6]])
    print("direction check:", s["counterfactual_direction_check"]); print("stability:", s["explanation_stability"])
    for c in s["validation"]:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    print("ALL CHECKS PASSED" if s["all_checks_passed"] else "SOME CHECKS FAILED")
    return 0 if s["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
