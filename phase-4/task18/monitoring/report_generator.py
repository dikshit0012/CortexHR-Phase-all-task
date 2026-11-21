"""Build, render and persist monitoring reports.

Outputs (default outputs/monitoring/, override CORTEXHR_MONITORING_DIR):
    latest_report.json       the last run, complete (the commit marker: written last)
    latest_report.md         the same as a readable document
    psi_results.csv          one row per feature, stable schema
    fairness_results.json    Task 12 and Task 15 sections
    monitoring_history.csv   one row per run, append-only, stable schema
    history/<run_id>/        the full report.json + psi_results.csv of EVERY run (never overwritten)

Guarantees
  * files are written to a temp name and os.replace()d, so a reader never sees a half-written file;
  * history is written BEFORE the latest_* files; if anything fails midway the previous complete "latest" set stays intact;
  * a monitoring run that crashes publishes an explicit FAILED / ALERT report (never a stale or invented OK);
  * no employee-level data, file path or secret is ever written - only aggregates.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import SCHEMA_VERSION
from . import monitoring_config as cfg
from .status import ALERT, ERROR

PSI_COLUMNS = ["run_id", "timestamp_utc", "feature", "kind", "method", "reference_n", "current_n", "reference_missing_pct",
               "current_missing_pct", "psi", "status", "reference_used", "training_reference_n", "psi_training_reference", "status_training_reference", "reference_internal_psi", "top_shift_bin", "top_shift_reference_pct", "top_shift_current_pct", "message"]
HISTORY_COLUMNS = ["run_id", "timestamp_utc", "execution_status", "status", "execution_ok", "evidence_complete", "features_checked",
                   "psi_ok", "psi_watch", "psi_alert", "psi_not_evaluated", "psi_error", "max_psi", "max_psi_feature", "twin_resume_status",
                   "twin_max_gap", "pay_gap_status", "reasons_count"]


def new_run_id(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%dT%H%M%S%f}Z-{uuid.uuid4().hex[:6]}"


def utc_now_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------------ build
def build_report(run_id: str, timestamp: str, validation: Dict[str, Any], drift: Dict[str, Any], twin: Dict[str, Any], pay: Dict[str, Any],
                 aggregate: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "timestamp_utc": timestamp, "execution_status": "COMPLETED",
        "status": aggregate["status"], "reasons": aggregate["reasons"], "reason_counts": aggregate["reason_counts"],
        "execution_ok": aggregate["execution_ok"], "evidence_complete": aggregate["evidence_complete"], "required_checks": list(required),
        "datasets": {"reference": {"name": cfg.REFERENCE_SHEET, "rows": validation.get("reference_rows")},
                     "scoring": {"name": cfg.SCORING_SHEET, "rows": validation.get("scoring_rows")}},
        "validation": validation, "drift": drift, "fairness": {"twin_resume": twin, "pay_gap": pay},
        "code_version": (os.environ.get("GITHUB_SHA") or "")[:12] or None,
    }


def build_failure_report(run_id: str, timestamp: str, code: str) -> Dict[str, Any]:
    """Published when the monitor itself could not complete. Never OK; carries a stable error code, not an exception text."""
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "timestamp_utc": timestamp, "execution_status": "FAILED", "status": ALERT,
        "reasons": [{"category": "execution_failure", "check": "monitor", "item": "run", "status": ERROR, "message": f"monitoring run failed: {code}"}],
        "reason_counts": {"execution_failure": 1}, "execution_ok": False, "evidence_complete": False, "required_checks": [],
        "datasets": {"reference": {"name": cfg.REFERENCE_SHEET, "rows": None}, "scoring": {"name": cfg.SCORING_SHEET, "rows": None}},
        "validation": {"status": ERROR, "message": "run failed before validation completed"},
        "drift": {"status": ERROR, "features_checked": 0, "counts": {}, "features": [], "max_psi": None, "max_psi_feature": None},
        "fairness": {"twin_resume": {"status": ERROR, "message": "not run"}, "pay_gap": {"status": ERROR, "message": "not run"}},
        "code_version": (os.environ.get("GITHUB_SHA") or "")[:12] or None,
    }


# ------------------------------------------------------------------------------------------------ rendering
def _fmt(v: Any, nd: int = 4) -> str:
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _pay_line(comp: Dict[str, Any]) -> str:
    m = (comp.get("metrics") or {}).get("predicted pay vs role band (ratio)")
    if comp.get("status") in ("NOT_EVALUATED", "ERROR") or not m:
        return f"{comp.get('status', 'NOT_EVALUATED')} ({comp.get('message', '')})"
    return f"{_fmt(m['current_gap'])} (baseline {_fmt(m['baseline_gap'])}) -> {comp['status']}"


def render_console(r: Dict[str, Any]) -> str:
    bar, thin = "=" * 50, "-" * 50
    d, f = r["drift"], r["fairness"]
    tw, pg = f["twin_resume"], f["pay_gap"]
    ref, sc = r["datasets"]["reference"], r["datasets"]["scoring"]
    L = [bar, "CORTEXHR - MODEL MONITORING REPORT", bar, "",
         f"Monitoring Timestamp: {r['timestamp_utc']}   (run {r['run_id']})",
         f"Reference Dataset:    {ref['name']} ({_fmt(ref['rows'])} rows)",
         f"Scoring Dataset:      {sc['name']} ({_fmt(sc['rows'])} rows)", f"Run:                  {r['execution_status']}", "",
         "DATASET VALIDATION", thin, f"{r['validation']['status']}: {r['validation'].get('message', '')}", "",
         "DRIFT MONITORING", thin, f"{'Feature':<34}{'PSI':>10}  {'Status':<14}{'Ref n':>7}{'Cur n':>7}  {'PSI vs training':>16}"]
    for x in d.get("features", []):
        L.append(f"{x['feature']:<34}{_fmt(x['psi'], 4):>10}  {x['status']:<14}{_fmt(x['reference_n']):>7}{_fmt(x['current_n']):>7}  {_fmt(x.get('psi_training_reference'), 4):>16}")
    c = d.get("counts", {})
    L += [f"Features checked: {d.get('features_checked', 0)}  OK={c.get('OK', 0)} WATCH={c.get('WATCH', 0)} ALERT={c.get('ALERT', 0)} "
          f"NOT_EVALUATED={c.get('NOT_EVALUATED', 0)} ERROR={c.get('ERROR', 0)}",
          "PSI < 0.10 OK | 0.10 <= PSI <= 0.25 WATCH | PSI > 0.25 ALERT",
          f"Reference used for PSI/Status: {(d.get('reference_policy') or {}).get('primary', 'n/a')} ({(d.get('reference_policy') or {}).get('matched_rows')} rows, "
          f"{(d.get('reference_policy') or {}).get('matched_rule')}); 'PSI vs training' = full reference ({(d.get('reference_policy') or {}).get('training_rows')} rows), always shown", "",
          "FAIRNESS MONITORING", thin, "Twin Resume:", f"  Status:          {tw['status']}  {tw.get('message', '')}",
          f"  Average Gap:     {_fmt(tw.get('average_gap'), 6)}", f"  Maximum Gap:     {_fmt(tw.get('maximum_gap'), 6)}",
          f"  Pairs Evaluated: {tw.get('valid_pairs', 0)} of {tw.get('expected_pairs', 'n/a')}", "",
          "Pay Gap:", f"  Status:          {pg['status']}  {pg.get('message', '')}"]
    comps = pg.get("components") or {}
    L += [f"  Gender Gap:      {_pay_line(comps['gender']) if 'gender' in comps else 'n/a'}",
          f"  Career Gap:      {_pay_line(comps['career_gap']) if 'career_gap' in comps else 'n/a'}",
          f"  Groups Evaluated: {pg.get('groups_evaluated', 0)}", "",
          "OVERALL STATUS", thin, f"STATUS: {r['status']}", f"Execution OK: {r['execution_ok']}   Evidence complete: {r['evidence_complete']}"]
    for x in r["reasons"]:
        L.append(f"  - [{x['category']}] {x['check']}/{x['item']}: {x['status']} {x['message']}".rstrip())
    L += ["", "PSI shows a change in input distributions, not model accuracy loss. Fairness gaps are monitoring signals, not proof of discrimination.", bar]
    return "\n".join(L)


def render_markdown(r: Dict[str, Any]) -> str:
    d, f = r["drift"], r["fairness"]
    tw, pg = f["twin_resume"], f["pay_gap"]
    ref, sc = r["datasets"]["reference"], r["datasets"]["scoring"]
    L = [f"# CortexHR model monitoring report", "", f"- **Overall status: {r['status']}**", f"- Timestamp (UTC): {r['timestamp_utc']}", f"- Run id: `{r['run_id']}`",
         f"- Execution: {r['execution_status']} (execution_ok={r['execution_ok']}, evidence_complete={r['evidence_complete']})",
         f"- Reference: `{ref['name']}` ({_fmt(ref['rows'])} rows); scoring: `{sc['name']}` ({_fmt(sc['rows'])} rows)", "",
         "## Drift (PSI)", "", "| Feature | Method | Ref n | Cur n | Ref missing % | Cur missing % | PSI | Status | PSI vs training (full) ref | Largest shift |", "|---|---|---:|---:|---:|---:|---:|---|---:|---|"]
    for x in d.get("features", []):
        shift = "" if x["top_shift_bin"] is None else f"{x['top_shift_bin']}: {x['top_shift_reference_pct']}% -> {x['top_shift_current_pct']}%"
        L.append(f"| {x['feature']} | {x['method'] or ''} | {_fmt(x['reference_n'])} | {_fmt(x['current_n'])} | {_fmt(x['reference_missing_pct'], 2)} | "
                 f"{_fmt(x['current_missing_pct'], 2)} | {_fmt(x['psi'], 4)} | {x['status']} | {_fmt(x.get('psi_training_reference'), 4)} | {shift} |")
    L += ["", "PSI < 0.10 OK; 0.10 <= PSI <= 0.25 WATCH; PSI > 0.25 ALERT. PSI measures a change in input distributions, not model accuracy.", "",
          "## Fairness", "", f"### Twin resume (Task 12) - {tw['status']}", "", f"{tw.get('message', '')}", "",
          f"- Screener: `{tw.get('screener')}`; valid pairs: {tw.get('valid_pairs', 0)} of {tw.get('expected_pairs')}",
          f"- Average gap: {_fmt(tw.get('average_gap'), 6)}; maximum gap: {_fmt(tw.get('maximum_gap'), 6)}; pairs above tolerance: {_fmt(tw.get('pairs_exceeding_tolerance'))}",
          f"- Baseline (Task 12): average {_fmt((tw.get('baseline') or {}).get('average_gap'), 6)}, maximum {_fmt((tw.get('baseline') or {}).get('maximum_gap'), 6)}", "",
          f"### Pay gap (Task 15) - {pg['status']}", "", f"{pg.get('message', '')}", ""]
    for name, comp in (pg.get("components") or {}).items():
        L.append(f"**{name}** - {comp['status']}: {comp.get('message', '')}")
        if comp.get("groups"):
            L.append("")
            L.append("| Group | n | Reliable (n >= %d) |" % cfg.PAY_MIN_GROUP)
            L.append("|---|---:|---|")
            L += [f"| {g['group']} | {g['n']} | {'yes' if g['reliable'] else 'no'} |" for g in comp["groups"]]
        if comp.get("metrics"):
            L += ["", "| Metric | Current gap | Baseline | Relative change | Status |", "|---|---:|---:|---:|---|"]
            L += [f"| {q} | {_fmt(m['current_gap'], 4)} | {_fmt(m['baseline_gap'], 4)} | {_fmt(m['relative_change'], 3)} | {m['status']} |" for q, m in comp["metrics"].items()]
        L.append("")
    L += ["Fairness gaps are monitoring signals, not proof of discrimination.", "", "## Findings", ""]
    L += [f"- **{x['category']}** - {x['check']}/{x['item']}: {x['status']}. {x['message']}" for x in r["reasons"]] or ["- none"]
    return "\n".join(L) + "\n"


def psi_rows(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"run_id": r["run_id"], "timestamp_utc": r["timestamp_utc"], **{k: x.get(k) for k in PSI_COLUMNS[2:]}} for x in r["drift"].get("features", [])]


def history_row(r: Dict[str, Any]) -> Dict[str, Any]:
    c = r["drift"].get("counts", {})
    tw, pg = r["fairness"]["twin_resume"], r["fairness"]["pay_gap"]
    return {"run_id": r["run_id"], "timestamp_utc": r["timestamp_utc"], "execution_status": r["execution_status"], "status": r["status"],
            "execution_ok": r["execution_ok"], "evidence_complete": r["evidence_complete"], "features_checked": r["drift"].get("features_checked", 0),
            "psi_ok": c.get("OK", 0), "psi_watch": c.get("WATCH", 0), "psi_alert": c.get("ALERT", 0), "psi_not_evaluated": c.get("NOT_EVALUATED", 0),
            "psi_error": c.get("ERROR", 0), "max_psi": r["drift"].get("max_psi"), "max_psi_feature": r["drift"].get("max_psi_feature"),
            "twin_resume_status": tw.get("status"), "twin_max_gap": tw.get("maximum_gap"), "pay_gap_status": pg.get("status"), "reasons_count": len(r["reasons"])}


def _csv_text(columns: List[str], rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return buf.getvalue()


# ------------------------------------------------------------------------------------------------ persistence
def sanitize(o: Any) -> Any:
    """Make a report strictly JSON-serialisable: numpy scalars -> Python numbers, NaN/inf -> None, tuples -> lists."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [sanitize(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)) and getattr(o, "shape", None) == ():     # numpy scalar
        o = o.item()
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _append_history(out: Path, r: Dict[str, Any]) -> None:
    path = out / "monitoring_history.csv"
    existing = ""
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        header = existing.splitlines()[0] if existing.strip() else ""
        if header != ",".join(HISTORY_COLUMNS):                     # schema changed: keep the old file, never overwrite history
            _atomic_write(out / f"monitoring_history.legacy-{r['run_id']}.csv", existing)
            existing = ""
    row_text = _csv_text(HISTORY_COLUMNS, [history_row(r)])
    _atomic_write(path, (existing if existing else row_text.splitlines()[0] + "\n") + row_text.split("\n", 1)[1])


def publish(r: Dict[str, Any], out_dir: Optional[Path] = None) -> Path:
    """Persist a completed (or FAILED) run. History first, then the latest_* files (json last = commit marker)."""
    out = Path(out_dir) if out_dir else cfg.output_dir()
    out.mkdir(parents=True, exist_ok=True)
    r = sanitize(r)
    payload = json.dumps(r, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n"      # allow_nan=False: JSON stays valid
    psi_csv = _csv_text(PSI_COLUMNS, psi_rows(r))
    run_dir = out / "history" / r["run_id"]
    if run_dir.exists():
        raise FileExistsError("history entry already exists")        # a run id is never reused: history is never overwritten
    _atomic_write(run_dir / "report.json", payload)
    _atomic_write(run_dir / "psi_results.csv", psi_csv)
    _append_history(out, r)
    fair = json.dumps({"run_id": r["run_id"], "timestamp_utc": r["timestamp_utc"], **r["fairness"]}, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n"
    _atomic_write(out / "fairness_results.json", fair)
    _atomic_write(out / "psi_results.csv", psi_csv)
    _atomic_write(out / "latest_report.md", render_markdown(r))
    _atomic_write(out / "latest_report.json", payload)
    return out
