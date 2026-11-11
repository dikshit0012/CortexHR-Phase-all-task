"""Turns service data into the official response_template text. Only the fields a template names are ever read, so salary,
net pay and every other column in a payslip row can never reach a reply. A template placeholder that cannot be filled means the
response is treated as invalid (the caller shows the official on_failure text) - nothing is invented or guessed.

FIELD NAMES: the template placeholders are official; the *service column names* behind them (e.g. `cl`, `present`, `manager_name`)
are not in any file I was given (mock_data/*.csv is missing), so each placeholder accepts a short list of likely column names.
UNVERIFIED against the real mock_data - see README 'Known limitations'."""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Callable, Dict, List, Mapping, Optional


def _pick(d: Mapping[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip() != "":
            return v
    return None


def _join(v: Any) -> Optional[str]:
    if v is None:
        return None
    return ", ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)


def fill(template: str, values: Mapping[str, Any]) -> Optional[str]:
    names = re.findall(r"\{(\w+)\}", template)
    if any(values.get(n) is None for n in names):
        return None
    return re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]), template)


def _leave_balance(d, s, ref):
    return {"cl": _pick(d, "cl", "casual_leave", "cl_balance", "casual"), "sl": _pick(d, "sl", "sick_leave", "sl_balance", "sick"),
            "el": _pick(d, "el", "earned_leave", "el_balance", "earned")}


def _leave_request(rows, s, ref):
    r = rows[0] if rows else None
    if r is None:
        return {}
    dates = _pick(r, "dates") or (f"{r['start_date']} to {r['end_date']}" if r.get("start_date") and r.get("end_date") else None)
    return {"dates": dates, "status": _pick(r, "status")}


def _holiday(rows, s, ref: date):
    best = None
    for r in rows:
        raw, name = _pick(r, "date", "holiday_date"), _pick(r, "name", "holiday_name", "holiday")
        try:
            dt = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        if dt >= ref and (best is None or dt < best[0]):
            best = (dt, name)
    return {"next_holiday": f"{best[1]} ({best[0].isoformat()})" if best and best[1] else None}


BUILDERS: Dict[str, Callable[[Any, Mapping[str, str], date], Dict[str, Any]]] = {
    "leave_balance": _leave_balance,
    "leave_request_status": _leave_request,
    "holiday_list": _holiday,
    "payslip_status": lambda d, s, ref: {"month": s.get("month"), "status": _pick(d, "status", "payslip_status")},
    "my_manager": lambda d, s, ref: {"manager_name": _pick(d, "manager_name", "manager", "manager_id")},
    "my_department": lambda d, s, ref: {"department_name": _pick(d, "department_name", "department", "dept_name", "department_id")},
    "team_members": lambda rows, s, ref: {"team_list": ", ".join(str(r.get("full_name") or r.get("employee_id")) for r in rows) or "no one reports to you"},
    "attendance_summary": lambda d, s, ref: {"month": s.get("month"), "present": _pick(d, "present", "present_days", "days_present"),
                                              "scheduled": _pick(d, "scheduled", "scheduled_days", "working_days"),
                                              "late": _pick(d, "late", "late_count", "late_arrivals", "late_days")},
    # --- /ai/v1 responses: UNVERIFIED shapes ---
    "unusual_attendance": lambda d, s, ref: {"count": _pick(d, "count", "n"), "days": _join(_pick(d, "days", "dates")) or ("none" if _pick(d, "count", "n") == 0 else None)},
    "salary_band_query": lambda d, s, ref: {"role": s.get("role"), "low": _pick(d, "low"), "high": _pick(d, "high"), "median": _pick(d, "median")},
    "attrition_risk_query": lambda d, s, ref: {"n": _pick(d, "n", "count", "high_risk_count"), "dept": s.get("department"), "drivers": _join(_pick(d, "drivers", "top_drivers"))},
    "headcount_forecast": lambda d, s, ref: {"month": _pick(d, "month", "period"), "value": _pick(d, "value", "forecast"), "low": _pick(d, "low"), "high": _pick(d, "high")},
    "payroll_flags": lambda d, s, ref: {"count": _pick(d, "count", "n"), "month": s.get("month"), "reason": _pick(d, "reason", "top_reason")},
    "resume_score_request": lambda d, s, ref: {"score": _pick(d, "score"), "role": s.get("role"), "matched": _join(_pick(d, "matched", "matched_skills")),
                                                "missing": _join(_pick(d, "missing", "missing_skills"))},
}


def render(intent: str, template: str, data: Any, slots: Mapping[str, str], ref: date) -> Optional[str]:
    builder = BUILDERS.get(intent)
    if builder is None:
        return None
    try:
        return fill(template, builder(data, slots, ref))
    except Exception:                                            # noqa: BLE001 - any shape surprise = invalid response, never a crash
        return None
