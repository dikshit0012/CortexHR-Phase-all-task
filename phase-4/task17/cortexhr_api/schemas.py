"""Pydantic schemas.

Attrition and pay-band request fields are exactly the columns the saved models were trained on (attrition-v2
`feature_list`, Task 15 `ALLOWED` features). Numeric bounds are sanity guards that reject impossible values; they are not
tuned to the training data. Missing numeric / categorical values are allowed because the saved pipelines impute them.
Anomaly request schemas come from the Task 8 service (see api/v1/anomaly.py).
"""
from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
# Optional categorical / numeric inputs default to None (= missing): the saved pipelines impute missing values.
Cat = Annotated[Optional[str], Field(default=None, min_length=1, max_length=64)]


def _num(lo: float, hi: float):
    return Annotated[Optional[float], Field(default=None, ge=lo, le=hi, allow_inf_nan=False)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------- common
class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# --------------------------------------------------------------------------------------- attrition
class AttritionRecord(Strict):
    """One employee-snapshot with the 33 features of attrition-v2 (no names, no identifiers, no exit information)."""
    record_ref: Optional[Label] = Field(default=None, description="Optional caller label, echoed back; never logged.")
    job_level: _num(1, 7)
    tenure_months: _num(0, 600)
    total_experience_years: _num(0, 60)
    previous_employers_count: _num(0, 40)
    commute_distance_km: _num(0, 500)
    annual_fixed_pay: _num(0, 1e9)
    comp_ratio: _num(0, 10)
    raises_received: _num(0, 80)
    last_raise_pct: _num(0, 300)
    months_since_last_raise: _num(0, 600)
    promotions_received: _num(0, 40)
    months_since_last_promotion: _num(0, 600)
    internal_transfers: _num(0, 40)
    last_performance_rating: _num(0, 10)
    last_engagement_score: _num(0, 10)
    manager_changes_12m: _num(0, 24)
    training_hours_12m: _num(0, 3000)
    attendance_rate_3m: _num(0, 1)
    late_arrivals_3m: _num(0, 100)
    avg_overtime_hours_per_month_3m: _num(0, 400)
    avg_hours_worked_3m: _num(0, 24)
    wfh_share_3m: _num(0, 1)
    unpaid_leave_days_6m: _num(0, 200)
    paid_leave_days_6m: _num(0, 200)
    attendance_rate_change_3m: _num(-1, 1)
    avg_unpaid_absence_days_3m: _num(0, 31)
    avg_variable_pay_3m: _num(0, 1e8)
    department_id: Cat
    role_id: Cat
    work_location: Cat
    work_mode: Cat
    employment_type: Cat
    highest_education: Cat


class AttritionRequest(Strict):
    records: List[AttritionRecord] = Field(min_length=1, description="1..max_batch records")


class AttritionResult(BaseModel):
    record_ref: Optional[str] = None
    attrition_probability_12m: float = Field(ge=0, le=1, description="Calibrated probability of leaving within 12 months")
    risk_band: Optional[Literal["Low", "Medium", "High"]] = Field(
        default=None, description="Official rank-based band (top 10% High, next 20% Medium). Only set when the request has >= 10 records.")


class AttritionResponse(BaseModel):
    model_version: str
    n_records: int
    risk_band_note: str
    results: List[AttritionResult]
    request_id: Optional[str] = None


# --------------------------------------------------------------------------------------- chat
class ChatRequest(Strict):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    session_id: Optional[Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]{1,64}$")]] = Field(
        default=None, description="Optional opaque id echoed back. The chatbot keeps no server-side conversation state.")


class ChatResponse(BaseModel):
    reply: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    fallback: bool
    fallback_reason: Optional[str] = None
    faq_ids: List[str] = []
    session_id: Optional[str] = None
    model_version: str
    request_id: Optional[str] = None


# --------------------------------------------------------------------------------------- resume ranking
class JobIn(Strict):
    job_id: Label = "JOB-1"
    job_title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    description_text: Annotated[str, StringConstraints(max_length=10_000)] = ""
    required_skills: Annotated[str, StringConstraints(max_length=1_000)] = Field(default="", description="Comma/semicolon separated")
    nice_to_have_skills: Annotated[str, StringConstraints(max_length=1_000)] = ""
    min_experience_years: _num(0, 60) = None


class ResumeIn(Strict):
    resume_id: Label
    resume_text: Annotated[str, StringConstraints(min_length=1, max_length=20_000)]


class ResumeRankRequest(Strict):
    job: JobIn
    resumes: List[ResumeIn] = Field(min_length=1, max_length=50)

    @field_validator("resumes")
    @classmethod
    def _unique(cls, v):
        ids = [r.resume_id for r in v]
        if len(ids) != len(set(ids)):
            raise ValueError("resume_id values must be unique")
        return v


class RankedResume(BaseModel):
    rank: int
    resume_id: str
    final_score: float
    tfidf_similarity: float
    skill_overlap: float
    matched_skills: List[str]
    missing_skills: List[str]
    preferred_skills_matched: List[str]
    experience_years: Optional[float] = None
    meets_min_experience: Optional[bool] = None


class ResumeRankResponse(BaseModel):
    method: str
    weights: Dict[str, float]
    fairness_note: str
    n_resumes: int
    ranking: List[RankedResume]
    request_id: Optional[str] = None


# --------------------------------------------------------------------------------------- pay band
class PayBandRequest(Strict):
    job_level: Annotated[int, Field(ge=1, le=7)]
    prior_experience_years: _num(0, 60)
    market_index: _num(0.1, 10)
    role_id: Cat
    department_id: Cat
    work_location: Cat
    highest_education: Cat


class PayBandResponse(BaseModel):
    model_version: str
    selected_model: str
    predicted_salary_mid: int
    recommended_salary_low: Optional[int] = None
    recommended_salary_high: Optional[int] = None
    band_status: str
    comparison: Dict[str, Optional[int]]
    features_used: List[str]
    fairness_note: str
    decision_note: str
    request_id: Optional[str] = None


# --------------------------------------------------------------------------------------- forecast
class ForecastRequest(Strict):
    department_id: Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9_\-]{1,16}$")] = "ALL"
    horizon_months: Annotated[int, Field(ge=1, le=6)] = 6


class ForecastPoint(BaseModel):
    month: date
    expected_headcount: float
    lower_80: float
    upper_80: float


class ForecastResponse(BaseModel):
    mode: Literal["live", "precomputed"]
    department_id: str
    trained_through: str
    model_version: str
    method: Optional[str] = None
    interval: float
    forecast: List[ForecastPoint]
    event_adjustments: List[Dict[str, Any]]
    event_adjustment_note: str
    note: str
    request_id: Optional[str] = None
