"""Display contracts: mandatory labels and form metadata. Field names and bounds come from contract_snapshot.json, generated
from the live API (scripts/sync_contract.py) and verified against it in tests_integration/."""
import json
from pathlib import Path

SNAPSHOT = json.loads((Path(__file__).parent / "contract_snapshot.json").read_text())

REVIEW_LABEL = "AI-generated, for human review"
LIMITATIONS = {
    "risk": "This score estimates risk. It does not establish that an employee will leave.",
    "anomaly": "Anomaly detection identifies unusual patterns. Human investigation is required. A flag is not proof of misconduct or payroll fraud.",
    "chat": "AI-generated responses may contain errors. Verify important HR decisions with an authorized human.",
    "resume": "Ranking is decision support and must not be used as the sole hiring criterion.",
    "payband": "This is a model estimate learned from past offers, which may contain past pay inequity. It is not a compensation decision.",
    "forecast": "The forecast is an estimate based on historical patterns and known workforce events. It is not a guaranteed outcome.",
    "monitoring": "Monitoring status is a health signal for the models; review flagged items with the data science owner.",
}

# --- Attrition form: grouped, using only fields that exist in the API schema (checked in tests) ---------------------------
ATTRITION_GROUPS = {
    "Role and pay": ["job_level", "annual_fixed_pay", "comp_ratio", "avg_variable_pay_3m", "department_id", "role_id", "employment_type"],
    "Career history": ["tenure_months", "total_experience_years", "previous_employers_count", "raises_received", "last_raise_pct",
                       "months_since_last_raise", "promotions_received", "months_since_last_promotion", "internal_transfers",
                       "manager_changes_12m", "highest_education"],
    "Performance and engagement": ["last_performance_rating", "last_engagement_score", "training_hours_12m"],
    "Attendance and hours": ["attendance_rate_3m", "attendance_rate_change_3m", "late_arrivals_3m", "avg_overtime_hours_per_month_3m",
                             "avg_hours_worked_3m", "wfh_share_3m", "avg_unpaid_absence_days_3m", "unpaid_leave_days_6m", "paid_leave_days_6m"],
    "Work set-up": ["commute_distance_km", "work_location", "work_mode"],
}
ATTRITION_FIELDS = {k: v for k, v in SNAPSHOT["attrition"]["fields"].items() if k != "record_ref"}
ATTRITION_SAMPLE = SNAPSHOT["attrition"]["sample"]

# --- Anomaly examples: the Task 8 service examples (deliberately unusual records) ----------------------------------------
ATTENDANCE_SAMPLE = {
    "attendance_feature_id": "AF-EX-0001", "employee_id": "EMP-0002", "month": "2026-10-01", "scheduled_days": 21, "present_days": 0,
    "half_days_unpaid": 0, "paid_leave_days": 0, "unpaid_absence_days": 21, "longest_unpaid_streak_days": 21, "late_arrivals": 0,
    "avg_hours_worked": 0.0, "overtime_hours": 0.0, "attendance_rate": 0.0, "wfh_days": 4, "job_level": 4, "department_id": "D08"}
PAYROLL_SAMPLE = {
    "payroll_item_id": "PI-EX-0001", "run_id": "PR-2026-10", "employee_id": "EMP-0002", "month": "2026-10-01", "role_id": "R047",
    "overtime_eligible": "N", "job_level": 4, "monthly_fixed_gross": 160000.0, "overtime_pay": 0.0, "variable_pay": 0.0,
    "gross_pay": 1600000.0, "pf_employee": 1800.0, "esi_employee": 0.0, "professional_tax": 200.0, "tds": 160000.0,
    "other_deductions": 0.0, "total_deductions": 162000.0, "net_pay": 1438000.0, "payable_days": 31.0, "lop_days": 0.0,
    "overtime_hours_paid": 0.0, "department_id": "D08", "variable_pay_type": None}
ANOMALY_FIELDS = {"attendance": SNAPSHOT["attendance"]["fields"], "payroll": SNAPSHOT["payroll"]["fields"]}
ANOMALY_SAMPLES = {"attendance": ATTENDANCE_SAMPLE, "payroll": PAYROLL_SAMPLE}

PAYBAND_FIELDS = SNAPSHOT["payband"]["fields"]
PAYBAND_SAMPLE = SNAPSHOT["payband"]["sample"]
FORECAST_DEPARTMENTS = SNAPSHOT["forecast_departments"]
RESUME_JOB_FIELDS = SNAPSHOT["resume_job"]["fields"]
