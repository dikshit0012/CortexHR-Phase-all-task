"""Workforce events: parsing, typing, historical effect estimates and transparent future adjustments.

Three kinds of numbers are kept apart on purpose:
  * statistically estimated - fitted from history (event dummies inside the forecasting model; descriptive hire/exit ratios);
  * event-based - taken from a known event window in the event table;
  * assumed - multipliers a person typed into data/future_events.csv; they are NOT estimated from data.

The shipped workforce_events sheet contains only past events (the latest ends 2026-04-30, before the last data month),
so no future adjustment is applied by default. Future events are supplied through an optional CSV (see FUTURE_COLUMNS).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loader import DataError

log = logging.getLogger(__name__)

# keyword rules (first match wins). Configurable: edit or pass your own list.
EVENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("hiring_freeze", ("freeze",)),
    ("hiring_drive", ("hiring drive", "scale-up", "build-out", "expansion", "hiring")),
]
POLICY = "policy_or_pay_cycle"          # no direct, documented headcount mechanism
EFFECT_DUMMIES = ["hiring_freeze", "hiring_drive"]
FUTURE_COLUMNS = ["event_id", "event_name", "start_date", "end_date", "department_id", "hires_multiplier", "exits_multiplier", "evidence"]


def classify_event(name: str, description: str = "", rules=EVENT_RULES) -> str:
    text = f"{name} {description}".lower()
    for label, keywords in rules:
        if any(k in text for k in keywords):
            return label
    return POLICY


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, derive duration and event type. Rejects windows that end before they start."""
    ev = events.copy()
    ev["start_date"] = pd.to_datetime(ev["start_date"])
    ev["end_date"] = pd.to_datetime(ev["end_date"])
    if (ev["end_date"] < ev["start_date"]).any():
        raise DataError("workforce_events contains an event that ends before it starts")
    ev["department_id"] = ev["department_id"].astype(str).str.strip()
    ev["event_type"] = [classify_event(n, d) for n, d in zip(ev["event_name"], ev["description"])]
    ev["first_month"] = ev["start_date"].dt.to_period("M").dt.to_timestamp()
    ev["last_month"] = ev["end_date"].dt.to_period("M").dt.to_timestamp()
    ev["duration_months"] = ((ev["last_month"].dt.year - ev["first_month"].dt.year) * 12
                             + ev["last_month"].dt.month - ev["first_month"].dt.month + 1)
    return ev


def _covers(ev_row, index: pd.DatetimeIndex) -> np.ndarray:
    return ((index >= ev_row["first_month"]) & (index <= ev_row["last_month"])).astype(float)


def build_calendar(events: pd.DataFrame, index: pd.DatetimeIndex, dept: str = config.COMPANY) -> pd.DataFrame:
    """0/1 dummies per month for the event types that have a headcount mechanism.

    A department series is affected by events tagged to it or to ALL. The company series (ALL) is affected by any
    freeze and by a hiring drive in any department (it is the sum of the departments).
    """
    cal = pd.DataFrame(0.0, index=index, columns=EFFECT_DUMMIES)
    for _, ev in events.iterrows():
        if ev["event_type"] not in EFFECT_DUMMIES:
            continue
        applies = dept == config.COMPANY or ev["department_id"] in (dept, config.COMPANY)
        if applies:
            cal[ev["event_type"]] = np.maximum(cal[ev["event_type"]].values, _covers(ev, index))
    return cal


def event_windows_mask(events: pd.DataFrame, index: pd.DatetimeIndex, dept: str) -> np.ndarray:
    """True for months touched by ANY event relevant to the series (used to build 'normal month' baselines)."""
    mask = np.zeros(len(index), dtype=bool)
    for _, ev in events.iterrows():
        if dept == config.COMPANY or ev["department_id"] in (dept, config.COMPANY):
            mask |= _covers(ev, index).astype(bool)
    return mask


def estimate_event_effects(history: pd.DataFrame, events: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """Descriptive effect of each historical event on monthly hires and exits.

    Compares the mean in the event months with the mean of the preceding `lookback` months that no event touched, in the
    same department. This is a before/after comparison on a handful of months - it is NOT a causal estimate.
    """
    rows = []
    for _, ev in events.iterrows():
        dept = ev["department_id"]
        g = history[history["department_id"] == dept].set_index("month").sort_index()
        in_event = (g.index >= ev["first_month"]) & (g.index <= ev["last_month"])
        pre = g.index < ev["first_month"]
        quiet = ~event_windows_mask(events, g.index, dept)
        base = g[pre & quiet].tail(lookback)
        ev_rows = g[in_event]
        row = {
            "event_id": ev["event_id"], "event_name": ev["event_name"], "event_type": ev["event_type"],
            "department_id": dept, "start_date": ev["start_date"].date(), "end_date": ev["end_date"].date(),
            "duration_months": int(ev["duration_months"]), "scope": "historical", "description": ev["description"],
            "n_event_months_observed": int(len(ev_rows)), "n_baseline_months": int(len(base)),
        }
        if len(ev_rows) and len(base):
            for col in ("hires", "exits"):
                b, e = float(base[col].mean()), float(ev_rows[col].mean())
                row[f"{col}_per_month_baseline"] = round(b, 2)
                row[f"{col}_per_month_event"] = round(e, 2)
                row[f"{col}_multiplier"] = round(e / b, 2) if b > 0 else np.nan
        used = ev["event_type"] in EFFECT_DUMMIES
        row["evidence"] = "statistically estimated (descriptive before/after; not causal)"
        row["used_in_model"] = ("event dummy in the advanced model's history" if used else "not modelled (no documented headcount mechanism)")
        row["applied_to_forecast"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def load_future_events(path: Path | None = None) -> pd.DataFrame:
    """Optional CSV of KNOWN future events with ASSUMED multipliers. Missing file = no known future events."""
    p = Path(path) if path else config.FUTURE_EVENTS_CSV
    if not p.is_file():
        return pd.DataFrame(columns=FUTURE_COLUMNS)
    fe = pd.read_csv(p)
    missing = [c for c in FUTURE_COLUMNS if c not in fe.columns]
    if missing:
        raise DataError(f"{p} is missing columns {missing}")
    return fe


def future_adjustments(history: pd.DataFrame, past_events: pd.DataFrame, future_events: pd.DataFrame,
                       months: pd.DatetimeIndex, departments: list[str]) -> pd.DataFrame:
    """Cumulative headcount adjustment per department and forecast month from assumed future-event multipliers.

    monthly effect = (hires_multiplier - 1) * normal_hires - (exits_multiplier - 1) * normal_exits
    where normal_* = mean of the last 12 months of the department's history not touched by a past event.
    ALL-scoped events apply to every real department; the company total is the sum of the department adjustments.
    Returns an empty frame when there are no future events. Historical events are never re-applied (no double counting).
    """
    cols = ["department_id", "month", "event_id", "monthly_effect", "cumulative_adjustment", "evidence"]
    if future_events is None or future_events.empty:
        return pd.DataFrame(columns=cols)
    real_depts = sorted(d for d in history["department_id"].unique() if d != config.COMPANY)
    pe = past_events if not past_events.empty else prepare_events(pd.DataFrame(columns=config.REQUIRED_EVENT_COLS))
    fe = future_events.copy()
    fe["start_date"], fe["end_date"] = pd.to_datetime(fe["start_date"]), pd.to_datetime(fe["end_date"])
    parts = []
    for dept in real_depts:
        g = history[history["department_id"] == dept].set_index("month").sort_index()
        quiet = ~event_windows_mask(pe, g.index, dept) if len(pe) else np.ones(len(g), dtype=bool)
        normal = g[quiet].tail(12)
        hires, exits = float(normal["hires"].mean()), float(normal["exits"].mean())
        effect = pd.Series(0.0, index=months)
        for _, ev in fe.iterrows():
            if ev["department_id"] not in (dept, config.COMPANY):
                continue
            inside = (months >= ev["start_date"].to_period("M").to_timestamp()) & (months <= ev["end_date"].to_period("M").to_timestamp())
            e = (ev["hires_multiplier"] - 1.0) * hires - (ev["exits_multiplier"] - 1.0) * exits
            effect[inside] += e
            for m in months[inside]:
                parts.append({"department_id": dept, "month": m, "event_id": ev["event_id"], "monthly_effect": e,
                              "evidence": ev.get("evidence", "assumed")})
    if not parts:
        return pd.DataFrame(columns=cols)
    adj = pd.DataFrame(parts)
    monthly = adj.groupby(["department_id", "month"])["monthly_effect"].sum().unstack(0).reindex(months).fillna(0.0)
    cum = monthly.cumsum()
    adj["cumulative_adjustment"] = [cum.loc[m, d] for d, m in zip(adj["department_id"], adj["month"])]
    total = cum.sum(axis=1)
    all_rows = pd.DataFrame({"department_id": config.COMPANY, "month": months, "event_id": "SUM_OF_DEPARTMENTS",
                             "monthly_effect": monthly.sum(axis=1).values, "cumulative_adjustment": total.values,
                             "evidence": "assumed (sum of department adjustments)"})
    return pd.concat([adj[cols], all_rows[cols]], ignore_index=True)


def apply_adjustments(fc: pd.DataFrame, adj: pd.DataFrame, dept: str) -> pd.DataFrame:
    """Shift mean and both bounds by the cumulative assumed adjustment for `dept`; keep headcount non-negative."""
    out = fc.copy()
    if adj is None or adj.empty:
        out["event_adjustment"] = 0.0
        return out
    a = adj[adj["department_id"] == dept].groupby("month")["cumulative_adjustment"].last()
    shift = a.reindex(out.index.union(a.index)).sort_index().ffill().reindex(out.index).fillna(0.0).values   # an event's effect persists after it ends
    for c in ("mean", "lower", "upper"):
        out[c] = np.maximum(out[c].values + shift, 0.0)
    out["event_adjustment"] = shift
    return out
