#!/usr/bin/env python
"""CortexHR model monitoring: dataset validation, eight-feature PSI drift, Task 12 and Task 15 fairness re-runs, one report.

    python monitor.py                       # run everything, write outputs/monitoring/, print the report
    python monitor.py --fail-on watch       # also exit non-zero on WATCH
    python monitor.py --checks drift        # run a subset (the others are reported NOT_EVALUATED, never OK)
    python monitor.py --no-write            # print only

Exit codes (so a scheduler can tell the cases apart):
    0  OK (or WATCH unless --fail-on watch)
    1  WATCH and --fail-on watch
    2  ALERT: drift, fairness, data-integrity or missing-evidence finding (see "reasons" in the report)
    3  the monitoring system itself failed (a check errored, or the run crashed)

Nothing is retrained and no model is changed by this script. See README.md for the alert response plan.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from monitoring import drift_monitor, fairness_monitor, monitoring_config as cfg, report_generator as rg, status as st  # noqa: E402

ALL_CHECKS = ("drift", "twin_resume", "pay_gap")
EXIT_OK, EXIT_WATCH, EXIT_ALERT, EXIT_FAILED = 0, 1, 2, 3


def _skipped(name: str) -> Dict[str, Any]:
    return {"status": st.NOT_EVALUATED, "message": f"{name} was not selected for this run"}


def _guard(fn, label: str) -> Dict[str, Any]:
    try:
        return fn()
    except Exception as exc:                                       # a check must never take the whole monitor down, nor look healthy
        return {"status": st.ERROR, "message": f"{label} failed unexpectedly ({type(exc).__name__})"}


def run_monitor(checks: Sequence[str] = ALL_CHECKS, out_dir: Optional[Path] = None, write: bool = True,
                now: Optional[datetime] = None) -> Tuple[Dict[str, Any], int]:
    now = now or datetime.now(timezone.utc)
    run_id, ts = rg.new_run_id(now), rg.utc_now_iso(now)
    try:
        # ---- 1. dataset validation
        frames: Dict[str, Any] = {"reference": None, "scoring": None}
        unavailable: Dict[str, str] = {}
        if "drift" in checks:
            for which, sheet, csv_env, wb_env in (("reference", cfg.REFERENCE_SHEET, "CORTEXHR_REFERENCE_CSV", "CORTEXHR_REFERENCE_WORKBOOK"),
                                                  ("scoring", cfg.SCORING_SHEET, "CORTEXHR_SCORING_CSV", "CORTEXHR_SCORING_WORKBOOK")):
                try:
                    frames[which] = drift_monitor.load_dataset(which, sheet, csv_env, wb_env)
                except drift_monitor.DatasetUnavailable as exc:
                    unavailable[which] = exc.code
            validation = _guard(lambda: drift_monitor.validate_datasets(frames["reference"], frames["scoring"], unavailable=unavailable), "dataset validation")
            validation.setdefault("reference_rows", None)
            validation.setdefault("scoring_rows", None)
            # ---- 2-3. eight-feature PSI
            reason = "; ".join(f"{k} dataset: {v}" for k, v in unavailable.items()) or "datasets not available"
            drift = _guard(lambda: drift_monitor.run_drift(frames["reference"], frames["scoring"], reason=reason), "drift monitoring")
            drift.setdefault("features", [])
        else:
            validation = {**_skipped("dataset validation"), "reference_rows": None, "scoring_rows": None}
            drift = {**_skipped("drift"), "features": [], "features_checked": 0, "counts": {}, "max_psi": None, "max_psi_feature": None}
        # ---- 4-5. fairness
        twin = _guard(fairness_monitor.evaluate_twin_resume, "twin-resume monitoring") if "twin_resume" in checks else _skipped("twin_resume")
        pay = _guard(fairness_monitor.evaluate_pay_gap, "pay-gap monitoring") if "pay_gap" in checks else _skipped("pay_gap")
        # ---- 6. overall status
        required = cfg.required_checks()
        agg = st.aggregate(st.collect_findings(validation, drift, twin, pay), required)
        report = rg.build_report(run_id, ts, validation, drift, twin, pay, agg, list(required))
    except Exception as exc:                                       # last resort: publish an explicit FAILED report, never a stale OK
        report = rg.build_failure_report(run_id, ts, type(exc).__name__)
    # ---- 7. persistence
    if write:
        try:
            rg.publish(report, out_dir)
        except Exception as exc:
            report["reasons"].append({"category": "execution_failure", "check": "persistence", "item": "outputs", "status": st.ERROR,
                                      "message": f"results could not be written ({type(exc).__name__})"})
            report.update(status=st.ALERT, execution_ok=False, evidence_complete=False)
            try:                                                    # best effort: leave an explicit FAILED marker instead of silence
                rg.publish(rg.build_failure_report(report["run_id"] + "-persist", ts, type(exc).__name__), out_dir)
            except Exception:
                pass
    return report, exit_code_for(report)


def exit_code_for(report: Dict[str, Any], fail_on: str = "alert") -> int:
    if report["execution_status"] != "COMPLETED" or not report["execution_ok"]:
        return EXIT_FAILED
    if report["status"] == st.ALERT:
        return EXIT_ALERT if fail_on != "never" or not report["evidence_complete"] else EXIT_OK
    if report["status"] == st.WATCH and fail_on == "watch":
        return EXIT_WATCH
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=None, help="default: outputs/monitoring (or CORTEXHR_MONITORING_DIR)")
    ap.add_argument("--fail-on", choices=("alert", "watch", "never"), default="alert")
    ap.add_argument("--checks", default=",".join(ALL_CHECKS), help="comma list of: " + ", ".join(ALL_CHECKS))
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    checks = tuple(c.strip() for c in a.checks.split(",") if c.strip())
    unknown = [c for c in checks if c not in ALL_CHECKS]
    if unknown:
        ap.error(f"unknown check(s): {', '.join(unknown)}")
    report, _ = run_monitor(checks, a.output_dir, write=not a.no_write)
    print(rg.render_console(report))
    return exit_code_for(report, a.fail_on)


if __name__ == "__main__":
    raise SystemExit(main())
