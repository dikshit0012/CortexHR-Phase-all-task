"""CortexHR Anomaly Service.  Run from the `task 8` folder:  uvicorn app.main:app

Endpoints
    GET  /v1/health                      liveness + whether the models are loaded
    GET  /v1/models                      model versions and limits (no paths, no data)
    POST /v1/anomaly/attendance          one attendance record  -> one official result row
    POST /v1/anomaly/payroll             one payroll record     -> one official result row
    POST /v1/anomaly/attendance/batch    1..max_batch_size attendance records -> results in input order
    POST /v1/anomaly/payroll/batch       1..max_batch_size payroll records    -> results in input order
    POST /v1/chat                        (Task 9) HR question -> intent, confidence, approved FAQ reply or safe fallback

Models and the reference workbook are loaded ONCE per process, in the lifespan handler. With several worker
processes every process holds its own copy in memory.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .bootstrap import load_service
from .chat import load_chat_service_safely, router as chat_router
from .config import Settings
from .schemas import (AnomalyResult, AttendanceBatchRequest, AttendanceRecord, BatchResponse, PayrollBatchRequest,
                      PayrollRecord)
from .scoring import AnomalyService

log = logging.getLogger("cortexhr.anomaly")
router = APIRouter(prefix="/v1")

_ERR_500 = {500: {"description": "Unexpected internal error while scoring (no details are exposed)."}}
_ERR_503 = {503: {"description": "Models are not loaded."}}


def get_service(request: Request) -> AnomalyService:
    service: Optional[AnomalyService] = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Scoring is unavailable: the models are not loaded.")
    return service


def _run(fn: Callable[[List[dict]], List[dict]], records: list) -> List[dict]:
    """Run a scoring function. Unexpected errors become a generic 500: no stack trace, no record values."""
    try:
        return fn([r.model_dump() for r in records])
    except Exception as exc:                                            # noqa: BLE001 - deliberate: never leak internals
        log.error("scoring failed (%s)", type(exc).__name__)            # type only: messages can contain data values
        raise HTTPException(status_code=500, detail="Internal error while scoring the request.") from None


def _check_batch(records: list, id_attr: str, limit: int) -> None:
    if len(records) > limit:
        raise HTTPException(422, detail=[{"loc": ["body", "records"], "type": "too_long",
                                          "msg": f"Batch too large: {len(records)} records, the maximum is {limit}."}])
    ids = [getattr(r, id_attr) for r in records]
    if len(set(ids)) != len(ids):
        raise HTTPException(422, detail=[{"loc": ["body", "records"], "type": "duplicate_record_id",
                                          "msg": f"{id_attr} values must be unique within one batch."}])


@router.get("/health", tags=["service"])
def health(request: Request):
    ready = getattr(request.app.state, "service", None) is not None
    return JSONResponse(status_code=200 if ready else 503,
                        content={"status": "ok" if ready else "unavailable", "service": "cortexhr-anomaly",
                                 "models_loaded": ready})


@router.get("/models", tags=["service"], responses=_ERR_503)
def models(service: AnomalyService = Depends(get_service)):
    return service.info()


@router.post("/anomaly/attendance", response_model=AnomalyResult, tags=["anomaly"], responses={**_ERR_500, **_ERR_503})
def anomaly_attendance(record: AttendanceRecord, service: AnomalyService = Depends(get_service)):
    return _run(service.score_attendance, [record])[0]


@router.post("/anomaly/payroll", response_model=AnomalyResult, tags=["anomaly"], responses={**_ERR_500, **_ERR_503})
def anomaly_payroll(record: PayrollRecord, service: AnomalyService = Depends(get_service)):
    return _run(service.score_payroll, [record])[0]


@router.post("/anomaly/attendance/batch", response_model=BatchResponse, tags=["anomaly"], responses={**_ERR_500, **_ERR_503})
def anomaly_attendance_batch(body: AttendanceBatchRequest, service: AnomalyService = Depends(get_service)):
    _check_batch(body.records, "attendance_feature_id", service.max_batch_size)
    results = _run(service.score_attendance, body.records)
    return {"detector": "attendance", "count": len(results), "results": results}


@router.post("/anomaly/payroll/batch", response_model=BatchResponse, tags=["anomaly"], responses={**_ERR_500, **_ERR_503})
def anomaly_payroll_batch(body: PayrollBatchRequest, service: AnomalyService = Depends(get_service)):
    _check_batch(body.records, "payroll_item_id", service.max_batch_size)
    results = _run(service.score_payroll, body.records)
    return {"detector": "payroll", "count": len(results), "results": results}


def create_app(settings: Optional[Settings] = None, service: Optional[AnomalyService] = None, chat_service: Optional[Any] = None) -> FastAPI:
    """App factory. `service` / `chat_service` let tests inject a ready service (or a test double) without touching production code."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: load everything once. A StartupError here stops the process with a clear message.
        app.state.service = service if service is not None else load_service(settings or Settings.from_env())
        log.info("models loaded: %s, %s", app.state.service.attendance.version, app.state.service.payroll.version)
        # Task 9: the chat model is loaded once too. If it is unavailable only /v1/chat answers 503; the anomaly endpoints are unaffected.
        app.state.chat = chat_service if chat_service is not None else load_chat_service_safely()
        yield
        app.state.service = None
        app.state.chat = None

    app = FastAPI(title="CortexHR Anomaly Service", version="1.0.0", lifespan=lifespan,
                  description="Attendance and payroll anomaly detection (Tasks 5-7) behind a validated API.")
    app.state.service = None
    app.state.chat = None
    app.include_router(router)
    app.include_router(chat_router)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        # FastAPI's default body echoes the submitted values (`input`); employee data must not be echoed back.
        errors = [{"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    return app


app = create_app()
