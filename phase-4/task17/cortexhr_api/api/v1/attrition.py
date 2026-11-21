from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Request

from ...registry import ATTRITION
from ...schemas import AttritionRequest, AttritionResponse, AttritionResult, ErrorResponse
from ...security import ApiError, require_api_key

router = APIRouter(prefix="/attrition", tags=["Attrition"], dependencies=[Depends(require_api_key)],
                   responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})

BAND_NOTE = ("The official risk band is rank-based (High = top 10% of scores, Medium = next 20%), so it is only meaningful "
             "for a batch of at least 10 records. For smaller requests risk_band is null; use the probability.")
EXAMPLE = {"records": [{"record_ref": "demo-1", "job_level": 3, "tenure_months": 30, "total_experience_years": 6.5,
    "previous_employers_count": 2, "commute_distance_km": 12.0, "annual_fixed_pay": 900000, "comp_ratio": 0.95,
    "raises_received": 2, "last_raise_pct": 6.0, "months_since_last_raise": 14, "promotions_received": 1,
    "months_since_last_promotion": 20, "internal_transfers": 0, "last_performance_rating": 3, "last_engagement_score": 3.4,
    "manager_changes_12m": 1, "training_hours_12m": 18, "attendance_rate_3m": 0.93, "late_arrivals_3m": 4,
    "avg_overtime_hours_per_month_3m": 8, "avg_hours_worked_3m": 8.4, "wfh_share_3m": 0.2, "unpaid_leave_days_6m": 1,
    "paid_leave_days_6m": 5, "attendance_rate_change_3m": -0.02, "avg_unpaid_absence_days_3m": 0.3,
    "avg_variable_pay_3m": 5000, "department_id": "D01", "role_id": "R001", "work_location": "Bengaluru",
    "work_mode": "Hybrid", "employment_type": "Full-time", "highest_education": "Bachelors"}]}


@router.post("/predict", response_model=AttritionResponse, summary="Probability of leaving within 12 months",
             openapi_extra={"requestBody": {"content": {"application/json": {"example": EXAMPLE}}}})
def predict(body: AttritionRequest, request: Request):
    """Scores each record with the saved **attrition-v2** model (calibrated logistic regression, Task 13). The model's own
    Task 2 preprocessing pipeline is part of the artifact, so raw feature values are sent as-is. Missing values are imputed
    by that pipeline. No threshold for a hard yes/no was defined in Tasks 13-14, so none is invented here.

    Errors: 401 (API key), 422 (validation), 503 (model not loaded)."""
    s, reg = request.app.state.settings, request.app.state.registry
    if len(body.records) > s.max_batch:
        raise ApiError(400, "batch_too_large", f"At most {s.max_batch} records per request.")
    art = reg.get(ATTRITION)
    feats = art["features"]
    rows = [r.model_dump(exclude={"record_ref"}) for r in body.records]
    X = pd.DataFrame(rows).reindex(columns=feats)
    X["never_promoted"] = (X["promotions_received"].fillna(0) <= 0).astype(int)
    X["never_raised"] = (X["raises_received"].fillna(0) <= 0).astype(int)
    for c in feats:
        if X[c].dtype == object and c not in ("department_id", "role_id", "work_location", "work_mode", "employment_type", "highest_education"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in ("department_id", "role_id", "work_location", "work_mode", "employment_type", "highest_education"):
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    p = art["model"].predict_proba(X)[:, 1]
    if p.shape != (len(X),) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ApiError(500, "invalid_model_output", "The model produced invalid probabilities.")
    bands = [None] * len(p)
    if len(p) >= 10:
        try:
            from cortexhr_task14.cortexhr_helpers import risk_bands
            bands = [str(b) for b in risk_bands(p)]
        except ImportError:
            order = np.argsort(-p)
            bands = ["Low"] * len(p)
            n_high = max(1, int(round(len(p) * 0.10)))
            n_medium = int(round(len(p) * 0.20))
            bands = np.asarray(bands, dtype=object)
            bands[order[:n_high]] = "High"
            bands[order[n_high:n_high + n_medium]] = "Medium"
            bands = bands.tolist()
    return AttritionResponse(
        model_version=art["metadata"]["version"], n_records=len(p), risk_band_note=BAND_NOTE, request_id=request.state.request_id,
        results=[AttritionResult(record_ref=r.record_ref, attrition_probability_12m=float(pi), risk_band=b)
                 for r, pi, b in zip(body.records, p, bands)])
