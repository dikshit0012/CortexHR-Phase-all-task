"""Paths (pathlib, environment overrides) and constants. No metric is stored here; every number is computed at run time."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: Path) -> Path:
    return Path(os.environ[name]).expanduser() if os.environ.get(name) else default


WORKBOOK = _env("CORTEXHR_WORKBOOK", ROOT / "data" / "CortexHR_Dataset.xlsx")          # master workbook (never committed)
SAMPLE_FORECAST_CSV = _env("CORTEXHR_SAMPLE_FORECAST", ROOT / "data" / "sample_forecast.csv")
FUTURE_EVENTS_CSV = _env("CORTEXHR_FUTURE_EVENTS", ROOT / "data" / "future_events.csv")  # optional, see event_handler.py
OUTPUTS = _env("CORTEXHR_TASK16_OUTPUTS", ROOT / "outputs")
CHARTS = _env("CORTEXHR_TASK16_CHARTS", ROOT / "charts")
REPORTS = _env("CORTEXHR_TASK16_REPORTS", ROOT / "reports")

HEADCOUNT_SHEET = "monthly_headcount_history"
EVENTS_SHEET = "workforce_events"
SAMPLE_SHEET = "sample_forecast"

COMPANY = "ALL"                      # company-total series
HORIZON = 6                          # months to forecast and to hold out
INTERVAL = 0.80                      # prediction-interval coverage
SEASONAL_PERIOD = 12
CV_MIN_TRAIN = 36                    # smallest training window used when selecting a model
CV_STEP = 3                          # months between rolling origins
SELECTION_TOLERANCE = 0.03           # simplest model within 3% of the best CV MAE wins
SEED = 42
MODEL_VERSION = "headcount-forecast-1.0"

REQUIRED_HEADCOUNT_COLS = [
    "month", "department_id", "headcount_end", "hires", "exits", "transfers_in", "transfers_out",
    "headcount_start", "attrition_rate_monthly_pct",
]
REQUIRED_EVENT_COLS = ["event_id", "start_date", "end_date", "department_id", "event_name", "description"]
SAMPLE_COLUMNS = [
    "forecast_id", "metric", "department_id", "forecast_month", "forecast_value", "lower_80", "upper_80",
    "method", "trained_through", "model_version",
]
