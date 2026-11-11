"""Pydantic request / response schemas.

REQUEST schemas are taken from the columns the existing detectors really need:
    AttendanceRecord = Task 5 `REQUIRED_COLUMNS` (+ the 3 optional workbook columns that are not model inputs)
    PayrollRecord    = Task 6 `REQUIRED_COLUMNS` (+ the 2 optional workbook columns that are not model inputs)
RESPONSE schema `AnomalyResult` = the 12 columns of the official `sample_anomaly_results` sheet (Task 7
`OFFICIAL_COLUMNS`), in the same order. The starter skeleton's `AttendanceIn` / `AnomalyOut` were smaller than
both and could not drive the real detectors or carry the official output, so they are replaced (see README).

Validation rules follow the data and the detector contracts, not guesses:
  * Numeric measures are REQUIRED KEYS but may be null: Task 5 imputes missing values and Task 6 has an explicit
    rule (INVALID_PAYROLL_DATA) for a missing amount, so a null is scored, not rejected. NaN / Infinity are rejected.
  * Money amounts may be negative (a negative net pay is exactly what the detector must flag).
  * Day / hour counts and the attendance rate cannot be negative; the rate is (present + 0.5 x half days) / scheduled.
  * Payroll month must be the first day of a calendar month and run_id must be PR-YYYY-MM (Task 6 hard requirements).
  * Unknown fields are rejected (`extra="forbid"`).
"""
from __future__ import annotations

from datetime import date
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

Key = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
Amount = Optional[Annotated[float, Field(allow_inf_nan=False)]]                         # may be negative / null
NonNeg = Optional[Annotated[float, Field(ge=0, allow_inf_nan=False)]]                   # hours, days
Count = Optional[Annotated[int, Field(ge=0)]]                                           # whole days / arrivals
Rate = Optional[Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]]


class AttendanceRecord(BaseModel):
    """One employee-month of attendance (sheet `attendance_monthly_features`)."""
    model_config = ConfigDict(extra="forbid")

    attendance_feature_id: Key
    employee_id: Key
    month: date
    scheduled_days: Count
    present_days: Count
    half_days_unpaid: Count
    paid_leave_days: Count
    unpaid_absence_days: Count
    longest_unpaid_streak_days: Count
    late_arrivals: Count
    avg_hours_worked: NonNeg
    overtime_hours: NonNeg
    attendance_rate: Rate
    # present in the workbook, not used by the detector
    wfh_days: Count = None
    job_level: Optional[int] = None
    department_id: Optional[Key] = None


class PayrollRecord(BaseModel):
    """One employee-month payslip (sheet `payroll_monthly_features`)."""
    model_config = ConfigDict(extra="forbid")

    payroll_item_id: Key
    run_id: Key
    employee_id: Key
    month: date
    role_id: Key
    overtime_eligible: Literal["Y", "N"]
    job_level: Optional[int]
    monthly_fixed_gross: Amount
    overtime_pay: Amount
    variable_pay: Amount
    gross_pay: Amount
    pf_employee: Amount
    esi_employee: Amount
    professional_tax: Amount
    tds: Amount
    other_deductions: Amount
    total_deductions: Amount
    net_pay: Amount                       # may be negative
    payable_days: NonNeg
    lop_days: NonNeg
    overtime_hours_paid: NonNeg
    # present in the workbook, not used by the detector
    department_id: Optional[Key] = None
    variable_pay_type: Optional[str] = Field(default=None, max_length=64)

    @field_validator("month")
    @classmethod
    def _first_of_month(cls, v: date) -> date:
        if v.day != 1:
            raise ValueError("month must be the first day of a calendar month (YYYY-MM-01)")
        return v

    @model_validator(mode="after")
    def _run_id_matches_month(self) -> "PayrollRecord":
        if self.run_id != f"PR-{self.month:%Y-%m}":
            raise ValueError("run_id must be PR-YYYY-MM of the payroll month")
        return self


class AttendanceBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[AttendanceRecord] = Field(min_length=1, description="1..max_batch_size records (see /v1/models)")


class PayrollBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[PayrollRecord] = Field(min_length=1, description="1..max_batch_size records (see /v1/models)")


class AnomalyResult(BaseModel):
    """The official output row (Task 7 `OFFICIAL_COLUMNS`). `flagged` is 'Y' / 'N'; for 'N' the reason and money are null."""
    result_id: str = Field(pattern=r"^AN-\d{4,}$")
    source_dataset: str
    record_id: str
    employee_id: str
    period: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    anomaly_score: float = Field(ge=0, le=1)
    flagged: Literal["Y", "N"]
    reason_code: Optional[str] = None
    reason_text: Optional[str] = None
    money_at_risk: Optional[float] = Field(default=None, ge=0)
    model_version: str
    scored_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


class BatchResponse(BaseModel):
    detector: Literal["attendance", "payroll"]
    count: int
    results: List[AnomalyResult]
