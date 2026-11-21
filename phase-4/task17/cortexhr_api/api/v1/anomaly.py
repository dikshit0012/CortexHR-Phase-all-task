from __future__ import annotations

import importlib
from typing import List, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...config import REPO_ROOT
from ...registry import ATTENDANCE, PAYROLL, load_task8_package
from ...schemas import ErrorResponse
from ...security import ApiError, require_api_key

try:
    load_task8_package(REPO_ROOT)
    _t8 = importlib.import_module("task8_app.schemas")
    AttendanceBatchRequest, PayrollBatchRequest, AnomalyResult = _t8.AttendanceBatchRequest, _t8.PayrollBatchRequest, _t8.AnomalyResult
except (FileNotFoundError, ImportError, ModuleNotFoundError):
    class AttendanceRecord(BaseModel):
        model_config = ConfigDict(extra="forbid")
        attendance_feature_id: str
        employee_id: str
        month: str
        scheduled_days: int = Field(ge=0)
        present_days: int = Field(ge=0)
        half_days_unpaid: int = Field(ge=0)
        paid_leave_days: int = Field(ge=0)
        unpaid_absence_days: int = Field(ge=0)
        longest_unpaid_streak_days: int = Field(ge=0)
        late_arrivals: int = Field(ge=0)
        avg_hours_worked: float = Field(ge=0)
        overtime_hours: float = Field(ge=0)
        attendance_rate: float = Field(ge=0, le=1)

    class PayrollRecord(BaseModel):
        model_config = ConfigDict(extra="forbid")
        payroll_item_id: str
        run_id: str = Field(pattern=r"^PR-\d{4}-\d{2}$")
        employee_id: str
        month: str
        role_id: str
        overtime_eligible: str
        job_level: int = Field(ge=1)
        monthly_fixed_gross: float = Field(ge=0)
        overtime_pay: float = Field(ge=0)
        variable_pay: float = Field(ge=0)
        gross_pay: float = Field(ge=0)
        pf_employee: float = Field(ge=0)
        esi_employee: float = Field(ge=0)
        professional_tax: float = Field(ge=0)
        tds: float = Field(ge=0)
        other_deductions: float = Field(ge=0)
        total_deductions: float = Field(ge=0)
        net_pay: float
        payable_days: float = Field(ge=0)
        lop_days: float = Field(ge=0)
        overtime_hours_paid: float = Field(ge=0)
        @model_validator(mode="after")
        def month_matches_run_id(self):
            expected = f"PR-{str(self.month)[:7]}"
            if self.run_id != expected:
                raise ValueError("run_id must match the record month")
            return self

    class AttendanceBatchRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        records: List[AttendanceRecord] = Field(min_length=1, max_length=100)

    class PayrollBatchRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        records: List[PayrollRecord] = Field(min_length=1, max_length=100)

    class AnomalyResult(BaseModel):
        model_config = ConfigDict(extra="forbid")


class AnomalyResponse(BaseModel):
    model_version: str
    n_records: int
    n_flagged: int
    results: List[AnomalyResult]
    note: str = ("Results are in input order; sort by result_id for the ranking (money at risk, reason priority, signals). "
                 "Scores use the reference scale stored in the artifact, so a record scores the same alone or in a batch.")
    request_id: Optional[str] = None


router = APIRouter(prefix="/anomaly", tags=["Anomaly Detection"], dependencies=[Depends(require_api_key)],
                   responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})


def _run(kind: str, body, request: Request) -> AnomalyResponse:
    reg = request.app.state.registry
    service = reg.get(kind)
    if len(body.records) > service.max_batch_size:
        raise ApiError(400, "batch_too_large", f"At most {service.max_batch_size} records per request.")
    recs = [r.model_dump() for r in body.records]
    try:
        out = service.score_attendance(recs) if kind == ATTENDANCE else service.score_payroll(recs)
    except ValueError as exc:                                                  # e.g. duplicate ids in one request
        raise ApiError(400, "invalid_records", str(exc)[:200]) from None
    version = reg.entries[kind].version or kind
    results = [AnomalyResult(**{k: r[k] for k in AnomalyResult.model_fields}) for r in out]
    return AnomalyResponse(model_version=version, n_records=len(results), n_flagged=sum(r.flagged == "Y" for r in results),
                           results=results, request_id=request.state.request_id)


@router.post("/attendance", response_model=AnomalyResponse, summary="Attendance anomaly detection")
def attendance(body: AttendanceBatchRequest, request: Request):
    """Runs the Task 5 detector (Isolation Forest + rules) and Task 7 explanation. The service needs the private reference
    history (CORTEXHR_WORKBOOK) to build month-over-month features; if it is not configured this returns **503**, never a guess."""
    return _run(ATTENDANCE, body, request)


@router.post("/payroll", response_model=AnomalyResponse, summary="Payroll anomaly detection with money at risk")
def payroll(body: PayrollBatchRequest, request: Request):
    """Runs the Task 6 detector (Isolation Forest + rules) and Task 7 reason codes and money-at-risk logic.
    Returns **503** when the reference history is not configured."""
    return _run(PAYROLL, body, request)
