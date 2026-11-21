from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, Request

from ...registry import PAYBAND
from ...schemas import ErrorResponse, PayBandRequest, PayBandResponse
from ...security import require_api_key

router = APIRouter(prefix="/payband", tags=["Pay-Band Analysis"], dependencies=[Depends(require_api_key)],
                   responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})

FAIRNESS_NOTE = ("Gender, age and career-gap are audit-only in Task 15 and are never model inputs. Group-gap metrics were computed "
                 "offline on a held-out set using protected-attribute data that is deliberately not available to this API, so they "
                 "cannot be returned per request.")


@router.post("/analyze", response_model=PayBandResponse, summary="Recommended pay for a role profile (decision support)",
             openapi_extra={"requestBody": {"content": {"application/json": {"example": {
                 "job_level": 3, "prior_experience_years": 5, "market_index": 1.2, "role_id": "R001",
                 "department_id": "D01", "work_location": "Bengaluru", "highest_education": "Bachelors"}}}}})
def analyze(body: PayBandRequest, request: Request):
    """Task 15 pay-band recommender. The CV-selected **gradient boosting** model predicts the midpoint from the seven
    allowed job attributes. The band (low/high) needs the Task 15 calibration ratios (CORTEXHR_PAYBAND_CALIBRATION); without them
    only the midpoint is returned and `band_status` says so. This is not a compensation decision."""
    art = request.app.state.registry.get(PAYBAND)
    cols = art["numeric"] + art["categorical"]
    X = pd.DataFrame([body.model_dump()])[cols]
    preds = {k: float(m.predict(X)[0]) for k, m in art["models"].items()}
    mid = max(preds["gradient_boosting"], 0.0)
    low = high = None
    status = "unavailable: calibration file not configured (midpoint only)"
    cal = art["calibration"]
    if cal:
        c = (cal.get("by_level") or {}).get(str(body.job_level)) or cal.get("global")
        if c:
            low, high = int(round(min(mid * c["low_ratio"], mid))), int(round(max(mid * c["high_ratio"], mid)))
            status = "calibrated (empirical 80% band from training out-of-fold residuals)"
    return PayBandResponse(
        model_version=request.app.state.registry.entries[PAYBAND].version, selected_model="gradient_boosting",
        predicted_salary_mid=int(round(mid)), recommended_salary_low=low, recommended_salary_high=high, band_status=status,
        comparison={"linear_regression": int(round(preds["linear_regression"])), "average_by_role_baseline": int(round(preds["average_by_role_baseline"]))},
        features_used=cols, fairness_note=FAIRNESS_NOTE, decision_note=art["note"], request_id=request.state.request_id)
