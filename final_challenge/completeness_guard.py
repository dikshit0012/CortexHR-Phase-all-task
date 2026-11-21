"""Final Challenge: input-completeness guard for attrition scoring.

Problem found in the Day 20 audit: the saved attrition-v2 pipeline median-imputes any missing feature, and the API accepts
records that omit the three monthly-derived features (the supplied attrition_scoring_2026.csv omits all three for every row).
The score then looks normal although part of the evidence was never seen. This module is a standalone, side-effect-free check
that tells a caller which records are complete enough to be ranked, and which must be routed to a human / completed first.
It does not touch any production code.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

# features that must be computed from monthly attendance/payroll before scoring (attrition-v2 metadata, "preprocessing")
MONTHLY_DERIVED = ("attendance_rate_change_3m", "avg_unpaid_absence_days_3m", "avg_variable_pay_3m")


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and v.strip() == "")


def check_record(record: Mapping[str, Any], feature_list: Sequence[str], critical: Iterable[str] = MONTHLY_DERIVED,
                 max_missing_share: float = 0.10) -> Dict[str, Any]:
    """Return {'status': 'complete'|'review', 'missing': [...], 'missing_critical': [...], 'missing_share': float}.

    A record needs review when ANY critical (monthly-derived) feature is missing, or when more than `max_missing_share`
    of all model features are missing. Nothing is imputed here and no score is produced."""
    miss = [f for f in feature_list if _missing(record.get(f))]
    crit = [f for f in miss if f in set(critical)]
    share = len(miss) / max(len(feature_list), 1)
    status = "review" if (crit or share > max_missing_share) else "complete"
    return {"status": status, "missing": miss, "missing_critical": crit, "missing_share": round(share, 4)}


def check_batch(records: Sequence[Mapping[str, Any]], feature_list: Sequence[str], **kw) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = [check_record(r, feature_list, **kw) for r in records]
    n_review = sum(r["status"] == "review" for r in rows)
    return {"n_records": len(rows), "n_complete": len(rows) - n_review, "n_review": n_review,
            "coverage": round((len(rows) - n_review) / len(rows), 4) if rows else 0.0, "records": rows}
