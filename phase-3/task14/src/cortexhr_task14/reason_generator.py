"""Plain-English reasons from SHAP contributions.

Every one of the 33 model features has an explicit entry in FEATURE_POLICY: a readable label, a theme, and whether it may be named in
a reason. Features that could act as proxies for protected characteristics (age, region, background; NOT measured here, no such data exists) or can reveal health / family
circumstances (leave) stay INSIDE the model (Task 14 cannot change attrition-v2) but are NEVER shown as a reason; their share of the
explanation is reported separately (suppressed_contribution) so the omission is visible, not hidden.

Reason rules
* only contributions >= MATERIALITY in the risk-RAISING direction are named first; risk-LOWERING factors fill empty slots and say so;
  if nothing else is notable the slot says so honestly (no invented reason).
* at most one reason per theme, so reasons never repeat or contradict each other (e.g. 'few raises' vs 'recent raise').
* wording is hedged ('is contributing to'), never says an employee will leave, never calls anyone a flight risk, uses no recommendation
  that implies discipline. Tenure always carries the neutral sentence required by the fairness rule.
* a missing value is only named when the gap itself is an HR fact (no promotion / no raise recorded); other missing values are
  data gaps, not findings, and are suppressed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config

HIGHER = "a higher-than-average predicted attrition risk"
LOWER = "a lower-than-average predicted attrition risk"
NO_FURTHER = "No further factor with a notable influence on this prediction was identified by the model."
TENURE_NOTE = "Length of service alone does not indicate an intention to leave."


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    theme: str
    eligible: bool = True
    exclusion_reason: str = ""
    kind: str = "numeric"                                       # numeric | categorical
    core: Optional[Callable[[object], str]] = None              # builds 'Recorded ... of 2.2 out of 5' from the value
    positioned: bool = True                                     # add the 'compared with other active employees' phrase
    missing_text: Optional[str] = None                          # set only when a missing value is itself an HR fact
    group_level: bool = False


def _n(v, d=0):
    return f"{float(v):,.{d}f}"


def _u(v, unit: str, d: int = 0) -> str:
    """'1 month' / '12 months' (correct singular for exactly 1)."""
    x = float(v)
    return f"{x:,.{d}f} {unit}" if abs(x - 1.0) < 1e-9 else f"{x:,.{d}f} {unit}s"


def _spec(name, label, theme, core=None, **kw):
    return FeatureSpec(name, label, theme, core=core, **kw)


def _excluded(name, label, theme, why, kind="numeric"):
    return FeatureSpec(name, label, theme, eligible=False, exclusion_reason=why, kind=kind)


_AGE = "possible proxy for age (experience accumulates over a working life; age is not in the data, so this is not measured); not cited as a precaution"
_LEAVE = "leave and absence can reveal health, family or religious circumstances; not cited as a precaution"
_REGION = "possible proxy for region / community background; not cited as a precaution"
_EDU = "possible proxy for socio-economic background; not cited as a precaution"

def _attendance_change(v) -> str:
    return f"Recorded change in monthly attendance over the last 3 months ({float(v) * 100:+.1f} percentage points)"


EXTREME_ATTENDANCE_CHANGE = 0.30
ATTENDANCE_CHECK_NOTE = " This is an unusually large change, so the underlying attendance records may be worth checking."


FEATURE_POLICY: Dict[str, FeatureSpec] = {s.name: s for s in [
    _spec("job_level", "Job level", "level_role", lambda v: f"Recorded job level ({_n(v)})"),
    _spec("tenure_months", "Length of service", "tenure", lambda v: f"Recorded length of service ({_u(v, 'month')})"),
    _excluded("total_experience_years", "Total experience", "experience", _AGE),
    _spec("previous_employers_count", "Previous employers", "career_history", lambda v: f"Number of previous employers on record ({_n(v)})"),
    _excluded("commute_distance_km", "Commute distance", "location", "reveals home location; possible proxy for region / community; not cited as a precaution"),
    _spec("annual_fixed_pay", "Annual fixed pay", "pay_level", lambda v: "Recorded annual fixed pay"),
    _spec("comp_ratio", "Pay compared with role mid-point", "pay_level", lambda v: f"Recorded pay of {float(v):.2f}x the role mid-point"),
    _spec("raises_received", "Pay revisions received", "raises", lambda v: f"Number of pay revisions recorded since joining ({_n(v)})"),
    _spec("last_raise_pct", "Size of latest raise", "raises", lambda v: f"Size of the most recent pay raise ({float(v):.1f}%)",
          missing_text="No pay raise is recorded since joining"),
    _spec("months_since_last_raise", "Time since last raise", "raises", lambda v: f"Time since the last recorded pay raise ({_u(v, 'month')})"),
    _spec("promotions_received", "Promotions received", "promotion", lambda v: f"Number of promotions recorded since joining ({_n(v)})"),
    _spec("months_since_last_promotion", "Time since last promotion", "promotion", lambda v: f"Time since the last recorded promotion ({_u(v, 'month')})",
          missing_text="No promotion is recorded since joining"),
    _spec("internal_transfers", "Internal transfers", "career_history", lambda v: f"Number of internal transfers on record ({_n(v)})"),
    _spec("last_performance_rating", "Latest performance rating", "performance", lambda v: f"Most recent performance rating ({float(v):.1f})"),
    _spec("last_engagement_score", "Latest engagement score", "engagement", lambda v: f"Most recent engagement survey score of {float(v):.1f} out of 5"),
    _spec("manager_changes_12m", "Manager changes", "career_history", lambda v: f"Number of manager changes in the last 12 months ({_n(v)})"),
    _spec("training_hours_12m", "Training hours", "development", lambda v: f"Recorded training in the last 12 months ({_u(v, 'hour')})"),
    _spec("attendance_rate_3m", "Attendance rate", "attendance", lambda v: f"Recorded attendance rate over the last 3 months ({float(v):.0%})"),
    _spec("late_arrivals_3m", "Late arrivals", "attendance", lambda v: f"Number of recorded late arrivals in the last 3 months ({_n(v)})"),
    _spec("avg_overtime_hours_per_month_3m", "Overtime hours", "workload", lambda v: f"Recorded overtime of {float(v):.1f} hours per month over the last 3 months"),
    _spec("avg_hours_worked_3m", "Daily hours worked", "workload", lambda v: f"Average recorded working time over the last 3 months ({float(v):.1f} hours a day)"),
    _spec("wfh_share_3m", "Work-from-home share", "work_arrangement", lambda v: f"Share of days worked from home over the last 3 months ({float(v):.0%})"),
    _excluded("unpaid_leave_days_6m", "Unpaid leave days", "leave", _LEAVE),
    _excluded("paid_leave_days_6m", "Paid leave days", "leave", _LEAVE),
    _spec("attendance_rate_change_3m", "Attendance trend", "attendance", lambda v: _attendance_change(v)),
    _excluded("avg_unpaid_absence_days_3m", "Unpaid absence days", "leave", _LEAVE),
    _spec("avg_variable_pay_3m", "Variable pay", "pay_level", lambda v: "Recorded average monthly variable pay"),
    _spec("department_id", "Department", "level_role", lambda v: f"The employee's department ({v})", kind="categorical", positioned=False, group_level=True),
    _spec("role_id", "Role", "level_role", lambda v: f"The employee's role group ({v})", kind="categorical", positioned=False, group_level=True),
    _excluded("work_location", "Office location", "location", _REGION, kind="categorical"),
    _spec("work_mode", "Work mode", "work_arrangement", lambda v: f"Recorded work mode ({v})", kind="categorical", positioned=False),
    _spec("employment_type", "Employment type", "work_arrangement", lambda v: f"Recorded employment type ({v})", kind="categorical", positioned=False),
    _excluded("highest_education", "Highest education", "education", _EDU, kind="categorical"),
]}

ELIGIBLE = [n for n, s in FEATURE_POLICY.items() if s.eligible]
EXCLUDED = [n for n, s in FEATURE_POLICY.items() if not s.eligible]


def policy_table() -> pd.DataFrame:
    return pd.DataFrame([{"feature": s.name, "label": s.label, "theme": s.theme, "named_in_reasons": s.eligible,
                          "exclusion_reason": s.exclusion_reason, "missing_value_named": bool(s.missing_text)} for s in FEATURE_POLICY.values()])


# --------------------------------------------------------------------------------------------------- wording
def position_phrase(value: float, population: pd.Series) -> Optional[str]:
    """Where the value sits among the active employees (factual, not a judgement). None if too few values or heavily tied."""
    pop = pd.to_numeric(population, errors="coerce").dropna()
    if len(pop) < 20 or pd.isna(value):
        return None
    if (pop == value).mean() > 0.40:
        return None
    pct = ((pop < value).sum() + 0.5 * (pop == value).sum()) / len(pop)
    if pct <= 0.25:
        return "on the lower side compared with other active employees"
    if pct >= 0.75:
        return "on the higher side compared with other active employees"
    return None                                           # typical values are not described: it would read as a contradiction


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or (not isinstance(v, str) and pd.isna(v))


def build_reason(spec: FeatureSpec, value, direction: int, population: Optional[pd.Series] = None) -> Optional[str]:
    """One plain-English sentence, or None if the feature must not be named (excluded / a missing value that is not an HR fact)."""
    if not spec.eligible or direction not in (1, -1):
        return None
    effect = HIGHER if direction > 0 else LOWER
    if _is_missing(value):
        return f"{spec.missing_text} and this is contributing to {effect}." if spec.missing_text else None
    if spec.name == "tenure_months":
        return (f"{spec.core(value)} is contributing to {'a higher-than-average' if direction > 0 else 'a lower-than-average'} estimate in this model. {TENURE_NOTE}")
    core = spec.core(value)
    if spec.name == "attendance_rate_change_3m" and abs(float(value)) >= EXTREME_ATTENDANCE_CHANGE:
        return f"{core} is contributing to {effect}.{ATTENDANCE_CHECK_NOTE}"       # extreme values may be data errors: no 'higher side' wording
    pos = position_phrase(float(value), population) if (spec.positioned and spec.kind == "numeric" and population is not None) else None
    core = f"{core}, which is {pos}," if pos else core
    if spec.group_level:
        return (f"{core} is associated with {effect} in the data the model learned from; this is a group-level pattern, not a finding about this individual.")
    return f"{core} is contributing to {effect}."


@dataclass
class Reason:
    feature: Optional[str]
    theme: Optional[str]
    contribution: float
    direction: str            # raises | lowers | none
    text: str


def select_reasons(contributions: pd.Series, values: pd.Series, population: pd.DataFrame, k: int = 3,
                   threshold: float = config.MATERIALITY, policy: Dict[str, FeatureSpec] = FEATURE_POLICY) -> Dict:
    """Pick up to k reasons from REAL contributions. Returns reasons (always k entries) and bookkeeping for the internal report."""
    unknown = [f for f in contributions.index if f not in policy]
    if unknown:
        raise KeyError(f"features without a wording policy: {unknown}")
    suppressed = float(sum(abs(contributions[f]) for f in contributions.index if not policy[f].eligible))
    gaps = [f for f in contributions.index if policy[f].eligible and _is_missing(values.get(f)) and not policy[f].missing_text]
    cands = [f for f in contributions.index if policy[f].eligible and f not in gaps]
    raising = sorted([f for f in cands if contributions[f] >= threshold], key=lambda f: -contributions[f])
    lowering = sorted([f for f in cands if contributions[f] <= -threshold], key=lambda f: contributions[f])
    chosen: List[Reason] = []
    used = set()
    n_raising = 0
    for group, sign in ((raising, 1), (lowering, -1)):
        for f in group:
            if len(chosen) == k:
                break
            s = policy[f]
            if s.theme in used:
                continue
            text = build_reason(s, values.get(f), sign, population[f] if f in population else None)
            if text is None:
                continue
            used.add(s.theme)
            chosen.append(Reason(f, s.theme, float(contributions[f]), "raises" if sign > 0 else "lowers", text))
            n_raising += sign > 0
    while len(chosen) < k:
        chosen.append(Reason(None, None, 0.0, "none", NO_FURTHER))
    return {"reasons": chosen, "n_risk_raising": n_raising, "n_risk_lowering": sum(r.direction == "lowers" for r in chosen),
            "n_empty": sum(r.direction == "none" for r in chosen), "suppressed_abs_contribution": suppressed, "data_gap_features": gaps}


# --------------------------------------------------------------------------------------------------- audit of the wording
FORBIDDEN_TERMS = [r"\bmale\b", r"\bfemale\b", r"\bwoman\b", r"\bman\b", r"\bgender\b", r"\bage\b", r"\baged\b", r"years old", r"\bhe\b", r"\bshe\b",
                   r"\bhis\b", r"\bher\b", r"religio", r"caste", r"\bmarried\b", r"\bmarital", r"pregnan", r"disab", r"ethnic", r"\brace\b",
                   r"nationalit", r"\bwill leave\b", r"\bwill quit\b", r"definitely", r"certain(ly)?\b", r"flight risk", r"\bguarantee",
                   r"disciplin", r"\bpunish", r"\bterminate", r"\bfire\b"]
_FORBIDDEN_RE = re.compile("|".join(FORBIDDEN_TERMS), re.I)


def audit_text(text: str, employee_names: Sequence[str] = ()) -> List[str]:
    """Problems found in one reason (empty list = clean). Checks protected / sensitive terms, certainty claims and employee names."""
    problems = [m.group(0) for m in _FORBIDDEN_RE.finditer(text)]
    problems += [f"name:{n}" for n in employee_names if n and re.search(rf"\b{re.escape(n)}\b", text, re.I)]
    return problems
