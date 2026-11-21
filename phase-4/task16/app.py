"""Vercel entrypoint for the Task 16 forecast API."""
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

FORECAST_PATH = Path(__file__).resolve().parent / "outputs" / "headcount_forecast.csv"

app = FastAPI(title="CortexHR Task 16 Forecast API", version="1.0.0")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "CortexHR Task 16 Forecast API",
        "health": "/health",
        "forecast": "/forecast",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/forecast")
def forecast() -> dict[str, list[dict[str, object]]]:
    if not FORECAST_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Forecast is unavailable. Run scripts/run_task16.py to generate it.",
        )

    return {"forecasts": pd.read_csv(FORECAST_PATH).to_dict(orient="records")}
