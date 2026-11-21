"""Overall status aggregation (OK / WATCH / ALERT) that keeps four different problems apart.

Per-check statuses
    OK             the check ran and is within its documented limits
    WATCH          the check ran and is in its warning range
    ALERT          the check ran and is beyond its alert threshold
    NOT_EVALUATED  the check could not run because evidence is missing (dataset / file / column absent)
    ERROR          the check crashed or was given invalid input (monitoring system failure)

Every non-OK finding carries one of four CATEGORIES, so an operator can tell them apart:
    drift               a PSI finding (the data changed)
    fairness            a twin-resume or pay-gap finding (a signal to review, not proof of discrimination)
    missing_evidence    something could not be evaluated because its data is not there
    execution_failure   the monitoring itself failed (exception, invalid values in a feature)
    data_integrity      dataset validation failed (empty or too-small dataset, monitored columns missing)

Overall status
    ALERT  if any ERROR; or any ALERT; or any NOT_EVALUATED on a REQUIRED check
    WATCH  if (not ALERT) and any WATCH; or any NOT_EVALUATED on an optional check
    OK     only if every finding is OK

A missing dataset or a failed fairness evaluation can therefore never produce OK.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

OK, WATCH, ALERT, NOT_EVALUATED, ERROR = "OK", "WATCH", "ALERT", "NOT_EVALUATED", "ERROR"
CHECK_STATUSES = (OK, WATCH, ALERT, NOT_EVALUATED, ERROR)
OVERALL_STATUSES = (OK, WATCH, ALERT)

_SEVERITY_ORDER = (ERROR, ALERT, NOT_EVALUATED, WATCH, OK)      # worst first, used to summarise one check made of several parts
_FAIRNESS_CHECKS = ("twin_resume", "pay_gap")


def worst(statuses: Iterable[str]) -> str:
    """Worst status of a group. Unknown values are treated as ERROR (fail closed). Empty group -> NOT_EVALUATED."""
    seen = set(statuses)
    if not seen:
        return NOT_EVALUATED
    for s in _SEVERITY_ORDER:
        if s in seen:
            return s
    return ERROR


def _category(check: str, status: str) -> str:
    if status == ERROR:
        return "execution_failure"
    if status == NOT_EVALUATED:
        return "missing_evidence"
    if check == "drift":
        return "drift"
    if check in _FAIRNESS_CHECKS:
        return "fairness"
    return "data_integrity"


def aggregate(findings: Sequence[Dict[str, Any]], required: Sequence[str]) -> Dict[str, Any]:
    """findings: dicts with keys check, item, status, message. Returns status, reasons, execution_ok, evidence_complete, category counts."""
    reasons: List[Dict[str, str]] = []
    level = 0                                                   # 0 OK, 1 WATCH, 2 ALERT
    for f in findings:
        st, chk = f.get("status"), f.get("check", "")
        if st == OK:
            continue
        if st not in CHECK_STATUSES:
            st = ERROR                                          # unknown status: fail closed
        if st == ERROR or st == ALERT:
            level = max(level, 2)
        elif st == NOT_EVALUATED:
            level = max(level, 2 if chk in required else 1)
        elif st == WATCH:
            level = max(level, 1)
        reasons.append({"category": _category(chk, st), "check": chk, "item": f.get("item", ""), "status": st,
                        "message": f.get("message", "")})
    grouped: Dict[tuple, Dict[str, str]] = {}                   # identical findings on several items (e.g. 8 features, same cause) -> one line
    for r in reasons:
        k = (r["category"], r["check"], r["status"], r["message"])
        if k in grouped:
            grouped[k]["item"] += ", " + r["item"]
        else:
            grouped[k] = dict(r)
    reasons = list(grouped.values())
    counts = Counter(r["category"] for r in reasons)
    return {
        "status": OVERALL_STATUSES[level],
        "reasons": reasons,
        "execution_ok": counts.get("execution_failure", 0) == 0,
        "evidence_complete": counts.get("missing_evidence", 0) == 0 and counts.get("execution_failure", 0) == 0,
        "reason_counts": dict(counts),
    }


def collect_findings(validation: Dict[str, Any], drift: Dict[str, Any], twin: Dict[str, Any], pay: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the four sections of a report into findings (one per feature / component) for aggregate()."""
    out: List[Dict[str, Any]] = [{"check": "dataset_validation", "item": "datasets", "status": validation["status"],
                                  "message": validation.get("message", "")}]
    feats = drift.get("features") or []
    if feats:
        for f in feats:
            out.append({"check": "drift", "item": f["feature"], "status": f["status"], "message": f.get("message", "")})
    else:
        out.append({"check": "drift", "item": "all_features", "status": drift["status"], "message": drift.get("message", "")})
    out.append({"check": "twin_resume", "item": "twin_resume", "status": twin["status"], "message": twin.get("message", "")})
    comps = pay.get("components") or {}
    if comps:
        for name, c in comps.items():
            out.append({"check": "pay_gap", "item": name, "status": c["status"], "message": c.get("message", "")})
    else:
        out.append({"check": "pay_gap", "item": "pay_gap", "status": pay["status"], "message": pay.get("message", "")})
    return out
