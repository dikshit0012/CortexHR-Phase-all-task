"""Headcount forecast service (Task 16).

LIVE mode    : the real Task 16 code (cleaning, event calendar, rolling-origin CV model selection, final fit) runs ONCE at
               startup on `monthly_headcount_history` (CORTEXHR_HEADCOUNT_CSV, a secret file). Requests only read the cached
               result, so nothing is retrained per request.
PRECOMPUTED  : if the history file is not available, the committed Task 16 output file is served and every response says
               `mode = "precomputed"` with `trained_through`. It is a fixed result, not a live prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import Settings

log = logging.getLogger("cortexhr.forecast")
REQUIRED_EVENT_COLS = ["event_id", "start_date", "end_date", "department_id", "event_name", "description"]


class ForecastService:
    def __init__(self, mode: str, version: str, trained_through: str, rows: pd.DataFrame, events: List[dict],
                 method: Dict[str, str], note: str, departments: List[str]):
        self.mode, self.version, self.trained_through = mode, version, trained_through
        self._rows, self.events, self.method, self.note, self.departments = rows, events, method, note, departments

    # ------------------------------------------------------------------ build (startup only)
    @classmethod
    def build(cls, s: Settings) -> "ForecastService":
        if s.headcount_csv is not None and s.headcount_csv.is_file():
            return cls._live(s)
        if s.headcount_csv is not None:
            log.warning("CORTEXHR_HEADCOUNT_CSV is set but the file does not exist; using the precomputed forecast")
        return cls._precomputed(s)

    @classmethod
    def _events_frame(cls, s: Settings) -> pd.DataFrame:
        if not s.events_csv.is_file():
            raise FileNotFoundError("workforce events file (Task 16 output) is missing")
        ev = pd.read_csv(s.events_csv)
        ev = ev[ev["event_id"].astype(str).str.upper() != "NONE"]            # placeholder row, not an event
        return ev[REQUIRED_EVENT_COLS].reset_index(drop=True)

    @classmethod
    def _live(cls, s: Settings) -> "ForecastService":
        from cortexhr_task16 import advanced_forecaster as af, config as c16, event_handler as eh, preprocessing as pp
        raw = pd.read_csv(s.headcount_csv)
        missing = [c for c in c16.REQUIRED_HEADCOUNT_COLS if c not in raw.columns]
        if missing:
            raise ValueError(f"headcount history is missing required columns: {missing}")
        hist = pp.clean_history(raw)
        quality = pp.validate_history(hist)
        events_raw = cls._events_frame(s)
        events = eh.prepare_events(events_raw)
        depts = list(dict.fromkeys(hist["department_id"].astype(str)))
        y_all, _ = pp.get_series(hist, c16.COMPANY)
        cal_all = eh.build_calendar(events, y_all.index, c16.COMPANY)
        cv = af.cross_validate(y_all, cal_all, c16.HORIZON)
        model = af.select_model(cv)                                           # selection exactly as Task 16 does it
        rows = []
        method: Dict[str, str] = {}
        for dept in depts:
            try:
                y, _ = pp.get_series(hist, dept)
                cal = eh.build_calendar(events, y.index, dept)
                try:
                    fc, used = af.fit_forecast(y, cal, model, c16.HORIZON), model
                except Exception:                                              # same documented fallback as Task 16
                    fc, used = af.fit_forecast(y, cal, "drift_events", c16.HORIZON), "drift_events"
            except Exception as exc:
                log.warning("department %s skipped: %s", dept, type(exc).__name__)
                continue
            method[dept] = af.SPECS[used].label
            for step, (month, r) in enumerate(fc.iterrows(), start=1):
                rows.append({"department_id": dept, "step": step, "forecast_month": month.strftime("%Y-%m-%d"),
                             "expected_headcount": round(float(r["mean"]), 1), "lower_80": round(float(min(r["lower"], r["mean"])), 1),
                             "upper_80": round(float(max(r["upper"], r["mean"])), 1)})
        last = y_all.index[-1].strftime("%Y-%m")
        df = pd.DataFrame(rows)
        ev_out = [{"event_id": r.event_id, "event_name": r.event_name, "department_id": r.department_id,
                   "start": str(r.start_date)[:10], "end": str(r.end_date)[:10], "event_type": r.event_type,
                   "applied_to_forecast": False} for r in events.itertuples()]
        note = (f"Live forecast fitted at startup on {quality['n_months']} months of history (to {last}). "
                f"Selected model: {af.SPECS[model].label}. No future workforce events are known, so none were applied.")
        return cls("live", c16.MODEL_VERSION, last, df, ev_out, method, note, sorted(set(df["department_id"])))

    @classmethod
    def _precomputed(cls, s: Settings) -> "ForecastService":
        if not s.precomputed_forecast.is_file():
            raise FileNotFoundError("neither the headcount history nor the precomputed Task 16 forecast is available")
        df = pd.read_csv(s.precomputed_forecast)
        df = df[df["metric"] == "headcount_end"].copy()
        df["step"] = df.groupby("department_id").cumcount() + 1
        df = df.rename(columns={"forecast_value": "expected_headcount"})
        method = df.groupby("department_id")["method"].first().to_dict()
        rows = df[["department_id", "step", "forecast_month", "expected_headcount", "lower_80", "upper_80"]].reset_index(drop=True)
        trained = str(df["trained_through"].iloc[0])
        version = str(df["model_version"].iloc[0])
        note = (f"PRECOMPUTED result from the committed Task 16 run (trained through {trained}). It is a fixed file, "
                "not a live prediction; supply CORTEXHR_HEADCOUNT_CSV to enable live forecasting.")
        events = []
        if s.events_csv.is_file():
            ev = cls._events_frame(s)
            events = [{"event_id": r.event_id, "event_name": r.event_name, "department_id": r.department_id,
                       "start": str(r.start_date)[:10], "end": str(r.end_date)[:10], "event_type": None,
                       "applied_to_forecast": False} for r in ev.itertuples()]
        return cls("precomputed", version, trained, rows, events, method, note, sorted(set(rows["department_id"])))

    # ------------------------------------------------------------------ request time (read only)
    def forecast(self, department_id: str, horizon: int) -> Optional[dict]:
        sub = self._rows[(self._rows["department_id"] == department_id) & (self._rows["step"] <= horizon)]
        if sub.empty:
            return None
        return {
            "mode": self.mode, "department_id": department_id, "trained_through": self.trained_through,
            "model_version": self.version, "method": self.method.get(department_id), "interval": 0.80,
            "forecast": [{"month": r.forecast_month, "expected_headcount": r.expected_headcount, "lower_80": r.lower_80,
                          "upper_80": r.upper_80} for r in sub.itertuples()],
            "event_adjustments": [e for e in self.events if e["department_id"] in ("ALL", department_id)],
            "event_adjustment_note": "Past hiring freezes and hiring drives are modelled as event dummies inside the model's history; "
                                     "no future events are known, so none are applied to the forecast months.",
            "note": self.note,
        }
