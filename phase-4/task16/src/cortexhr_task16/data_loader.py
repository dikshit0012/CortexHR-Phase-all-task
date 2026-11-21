"""Load the three Task 16 inputs from the real workbook. Column names are validated, never invented."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)


class DataError(ValueError):
    """Raised when an input is missing or does not have the documented structure."""


def _read_sheet(sheet: str, workbook: Path | None) -> pd.DataFrame:
    path = Path(workbook) if workbook else config.WORKBOOK
    if not path.is_file():
        raise DataError(
            f"Workbook not found: {path}. Set CORTEXHR_WORKBOOK to the CortexHR master workbook "
            f"(it must contain the sheets {config.HEADCOUNT_SHEET}, {config.EVENTS_SHEET}, {config.SAMPLE_SHEET})."
        )
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except ValueError as exc:  # missing sheet
        raise DataError(f"Sheet '{sheet}' not found in {path}: {exc}") from exc


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f"{name} is missing required columns: {missing}")


def load_headcount_history(workbook: Path | None = None) -> pd.DataFrame:
    df = _read_sheet(config.HEADCOUNT_SHEET, workbook)
    _require(df, config.REQUIRED_HEADCOUNT_COLS, config.HEADCOUNT_SHEET)
    log.info("Loaded %s: %d rows x %d columns", config.HEADCOUNT_SHEET, *df.shape)
    return df


def load_workforce_events(workbook: Path | None = None) -> pd.DataFrame:
    df = _read_sheet(config.EVENTS_SHEET, workbook)
    _require(df, config.REQUIRED_EVENT_COLS, config.EVENTS_SHEET)
    log.info("Loaded %s: %d events", config.EVENTS_SHEET, len(df))
    return df


def load_sample_forecast(csv_path: Path | None = None, workbook: Path | None = None) -> pd.DataFrame:
    """The sample file is the output contract. Prefer the CSV (exact text format), fall back to the sheet."""
    path = Path(csv_path) if csv_path else config.SAMPLE_FORECAST_CSV
    if path.is_file():
        df = pd.read_csv(path)
    else:
        df = _read_sheet(config.SAMPLE_SHEET, workbook)
        df["forecast_month"] = pd.to_datetime(df["forecast_month"]).dt.strftime("%Y-%m-%d")
    _require(df, config.SAMPLE_COLUMNS, "sample_forecast")
    return df
