from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ...registry import FORECAST
from ...schemas import ErrorResponse, ForecastRequest, ForecastResponse
from ...security import ApiError, require_api_key

router = APIRouter(prefix="/forecast", tags=["Headcount Forecast"], dependencies=[Depends(require_api_key)],
                   responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})


@router.post("/headcount", response_model=ForecastResponse, summary="Month-end headcount forecast with 80% interval",
             openapi_extra={"requestBody": {"content": {"application/json": {"example": {"department_id": "ALL", "horizon_months": 6}}}}})
def headcount(body: ForecastRequest, request: Request):
    """Task 16 forecast. `mode = live` means the model was fitted at server startup on the real history; `mode = precomputed`
    means a fixed Task 16 result is being served because the history file is not available to this server. Nothing is
    retrained during a request. The forecast covers up to 6 months after `trained_through`."""
    svc = request.app.state.registry.get(FORECAST)
    out = svc.forecast(body.department_id, body.horizon_months)
    if out is None:
        raise ApiError(404, "unknown_department", f"No forecast exists for department '{body.department_id}'. Available: {', '.join(svc.departments)}.")
    out["request_id"] = request.state.request_id
    return out
