"""Health (liveness), model availability and readiness. Public: they expose no employee data or paths."""
from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...config import API_VERSION
from ...registry import ALL_NAMES, ATTRITION, CHATBOT, FORECAST, PAYBAND, RESUME

router = APIRouter(tags=["Health"])

# Capabilities that must be loaded for the service to call itself ready. Anomaly detectors need the private reference
# workbook, so they are reported separately and do not block readiness of the other endpoints.
CORE = (ATTRITION, CHATBOT, RESUME, PAYBAND, FORECAST)


@router.get("/health", summary="Application health and model availability")
def health(request: Request):
    """Three separate facts: `application` (the process is up), `models` (what actually loaded) and `ready`
    (can predictions be served). `models_available` never lists a model that failed to load."""
    reg, started = request.app.state.registry, request.app.state.started_at
    loaded = [n for n in ALL_NAMES if reg.entries[n].loaded]
    missing = [n for n in ALL_NAMES if n not in loaded]
    return {
        "application": "up", "api_version": API_VERSION, "uptime_seconds": round(time.time() - started, 1),
        "environment": request.app.state.settings.environment,
        "models_available": loaded, "models_unavailable": missing,
        "all_models_loaded": not missing,
        "models": {n: {"status": reg.entries[n].status, "version": reg.entries[n].version, "mode": reg.entries[n].mode,
                       "detail": reg.entries[n].detail, "trained_with": reg.entries[n].trained_with,
                       "running_with": reg.entries[n].running_with, "compatibility": reg.entries[n].compatibility}
                   for n in ALL_NAMES},
        "forecast_mode": reg.entries[FORECAST].mode if reg.entries[FORECAST].loaded else None,
    }


@router.get("/ready", summary="Readiness: can the API serve predictions?")
def ready(request: Request):
    """200 only when every core capability is loaded. 503 otherwise, with the list of what is missing.
    Anomaly detectors are optional for readiness but listed in `degraded`."""
    reg = request.app.state.registry
    blocking = [n for n in CORE if not reg.entries[n].loaded]
    degraded = [n for n in ALL_NAMES if n not in CORE and not reg.entries[n].loaded]
    body = {"ready": not blocking, "blocking": blocking, "degraded": degraded,
            "auth_configured": (not request.app.state.settings.require_auth) or bool(request.app.state.settings.api_keys)}
    if not body["auth_configured"]:
        body["ready"] = False
        body["blocking"] = blocking + ["authentication"]
    return JSONResponse(body, status_code=200 if body["ready"] else 503)
