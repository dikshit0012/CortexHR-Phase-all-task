"""Hand-written data dictionary for the `attrition_dataset` sheet (Task 1, step 2).

Each entry: column -> (kind, role, plain-English meaning, audit note)
  kind: id | date | category | number | count | share | label
  role: ID | FEATURE | LABEL | LEAK | PROVENANCE
"""
import pandas as pd

D = {
    # --- identity / provenance -------------------------------------------------------------
    "row_id": ("id", "ID", "Running row number in the sheet.", "Meaningless for modelling; never a feature."),
    "employee_id": ("id", "ID", "Employee code. EMP-xxxx = real employee, HIST-xxxx = simulated history.", "Use as the GROUP key when splitting data so one person is never in train and test."),
    "observation_date": ("date", "PROVENANCE", "The day the features were measured (30 Sep 2025 for real staff, 2021-2025 for history).", "Use for time-aware splits, not as a feature."),
    "source": ("category", "PROVENANCE", "Where the row came from: real snapshot or historical extension.", "Describes data origin, not the employee. Drop from features."),
    # --- who the person is -----------------------------------------------------------------
    "department_id": ("category", "FEATURE", "Department code D00-D10 (D01 Engineering ... D10 Legal).", "D00 Executive never appears in this table."),
    "role_id": ("category", "FEATURE", "Job role code (48 distinct roles).", "High-cardinality: encode carefully."),
    "job_level": ("count", "FEATURE", "Seniority level 1 (associate) to 6 (head).", "Level 6 has only 2 rows."),
    "work_location": ("category", "FEATURE", "Office city.", "13 rows use messy spellings (BENGALURU, Madras ...)."),
    "work_mode": ("category", "FEATURE", "Onsite, Hybrid or Remote.", ""),
    "employment_type": ("category", "FEATURE", "Permanent or fixed-term contract.", "Contract staff are only 6% of rows."),
    "highest_education": ("category", "FEATURE", "Highest qualification, 27 free-text values.", "15 blanks. Many near-duplicate labels (B.E. vs B.Tech)."),
    # --- experience -----------------------------------------------------------------------
    "tenure_months": ("count", "FEATURE", "Months with the company at the observation date.", "4 impossible negative values."),
    "total_experience_years": ("number", "FEATURE", "Years of work in total: earlier jobs plus time here.", "Some values are negative or smaller than tenure."),
    "previous_employers_count": ("count", "FEATURE", "How many employers the person had before joining.", ""),
    "commute_distance_km": ("number", "FEATURE", "Home-to-office distance in km.", "55 blanks."),
    # --- pay & career ---------------------------------------------------------------------
    "annual_fixed_pay": ("number", "FEATURE", "Yearly fixed salary in INR.", "Right-skewed; 3 values are 10x too large (typos)."),
    "comp_ratio": ("number", "FEATURE", "Pay divided by the role's mid-point; below 1 = underpaid.", "Useful cross-check for pay typos."),
    "raises_received": ("count", "FEATURE", "Number of pay revisions since joining.", ""),
    "last_raise_pct": ("number", "FEATURE", "Size of the most recent raise (%).", "Blank = never had a raise (structural)."),
    "months_since_last_raise": ("count", "FEATURE", "Months since the last raise (or since joining if none).", ""),
    "promotions_received": ("count", "FEATURE", "Number of promotions since joining.", ""),
    "months_since_last_promotion": ("number", "FEATURE", "Months since the last promotion.", "Blank = never promoted (structural, 61%)."),
    "internal_transfers": ("count", "FEATURE", "Moves between departments.", ""),
    # --- performance & sentiment ----------------------------------------------------------
    "last_performance_rating": ("count", "FEATURE", "Latest appraisal rating, 1-5.", "Blank = joined under 12 months ago, no appraisal yet."),
    "last_engagement_score": ("number", "FEATURE", "Latest engagement survey score, 1-5.", "Blank = did not respond (22%)."),
    "manager_changes_12m": ("count", "FEATURE", "Times the manager changed in the last year.", ""),
    "training_hours_12m": ("number", "FEATURE", "Training hours in the last year.", "Long right tail, plausible."),
    # --- attendance & working pattern -----------------------------------------------------
    "attendance_rate_3m": ("share", "FEATURE", "Share of scheduled days attended, last 3 months (0-1).", "2 values above 1."),
    "late_arrivals_3m": ("count", "FEATURE", "Late check-ins in the last 3 months.", ""),
    "avg_overtime_hours_per_month_3m": ("number", "FEATURE", "Average monthly overtime hours.", "75% are zero; max 50."),
    "avg_hours_worked_3m": ("number", "FEATURE", "Average hours per working day.", "One implausible low value (1.8h)."),
    "wfh_share_3m": ("share", "FEATURE", "Share of days worked from home (0-1).", "Many values slightly below 0 or above 1."),
    "unpaid_leave_days_6m": ("number", "FEATURE", "Unpaid leave days, last 6 months.", ""),
    "paid_leave_days_6m": ("count", "FEATURE", "Paid leave days, last 6 months.", ""),
    "dept_attrition_rate_prior_12m": ("share", "FEATURE", "Share of the department that left in the 12 months BEFORE the snapshot.", "Legitimate (looks backwards), but confirm with the Python team."),
    # --- label ----------------------------------------------------------------------------
    "left_within_12m": ("label", "LABEL", "TARGET: 1 if the person left within 12 months after the observation date.", "14.6% are 1."),
    "voluntary_left_within_12m": ("label", "LEAK", "1 if they resigned (rather than being let go). A sub-type of the target.", "Known only once someone has left."),
    # --- post-outcome info ----------------------------------------------------------------
    "exit_date": ("date", "LEAK", "Last working day. Blank for people who stayed.", "Filled for exactly the rows where target = 1."),
    "resignation_date": ("date", "LEAK", "Day the resignation was submitted.", "Only exists for people who resigned."),
    "exit_reason": ("category", "LEAK", "Reason recorded when leaving.", "Only exists for leavers."),
    "exit_interview_done": ("category", "LEAK", "Y/N: was an exit interview held.", "Only exists for leavers."),
    "fnf_settlement_amount": ("number", "LEAK", "Full-and-final settlement paid at exit.", "Only exists for leavers."),
}


def dictionary_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Dictionary joined with measured facts (dtype, missing %, distinct values, example)."""
    rows = []
    for col in df.columns:
        kind, role, meaning, note = D[col]
        s = df[col]
        ex = s.dropna().iloc[0] if s.notna().any() else ""
        rows.append({
            "column": col, "role": role, "kind": kind, "pandas_dtype": str(s.dtype),
            "missing_n": int(s.isna().sum()), "missing_pct": round(100 * s.isna().mean(), 1),
            "distinct": int(s.nunique()), "example": str(ex)[:28],
            "meaning": meaning, "audit_note": note,
        })
    return pd.DataFrame(rows)
