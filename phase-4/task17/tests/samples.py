"""Safe synthetic sample inputs shaped like the real schemas (no real employee data)."""
ATTRITION = {"record_ref": "demo-1", "job_level": 3, "tenure_months": 30, "total_experience_years": 6.5, "previous_employers_count": 2,
    "commute_distance_km": 12.0, "annual_fixed_pay": 900000, "comp_ratio": 0.95, "raises_received": 2, "last_raise_pct": 6.0,
    "months_since_last_raise": 14, "promotions_received": 1, "months_since_last_promotion": 20, "internal_transfers": 0,
    "last_performance_rating": 3, "last_engagement_score": 3.4, "manager_changes_12m": 1, "training_hours_12m": 18,
    "attendance_rate_3m": 0.93, "late_arrivals_3m": 4, "avg_overtime_hours_per_month_3m": 8, "avg_hours_worked_3m": 8.4,
    "wfh_share_3m": 0.2, "unpaid_leave_days_6m": 1, "paid_leave_days_6m": 5, "attendance_rate_change_3m": -0.02,
    "avg_unpaid_absence_days_3m": 0.3, "avg_variable_pay_3m": 5000, "department_id": "D01", "role_id": "R001",
    "work_location": "Bengaluru", "work_mode": "Hybrid", "employment_type": "Full-time", "highest_education": "Bachelors"}
ATTENDANCE = {"attendance_feature_id": "AF-TEST-1", "employee_id": "E-TEST-1", "month": "2026-09-01", "scheduled_days": 22, "present_days": 20,
    "half_days_unpaid": 0, "paid_leave_days": 2, "unpaid_absence_days": 0, "longest_unpaid_streak_days": 0, "late_arrivals": 1,
    "avg_hours_worked": 8.2, "overtime_hours": 3.0, "attendance_rate": 0.91}
PAYROLL = {"payroll_item_id": "PI-TEST-1", "run_id": "PR-2026-09", "employee_id": "E-TEST-1", "month": "2026-09-01", "role_id": "R001",
    "overtime_eligible": "Y", "job_level": 3, "monthly_fixed_gross": 75000.0, "overtime_pay": 2000.0, "variable_pay": 5000.0,
    "gross_pay": 82000.0, "pf_employee": 9000.0, "esi_employee": 0.0, "professional_tax": 200.0, "tds": 4000.0, "other_deductions": 0.0,
    "total_deductions": 13200.0, "net_pay": 68800.0, "payable_days": 30.0, "lop_days": 0.0, "overtime_hours_paid": 8.0}
CHAT = {"message": "What is the notice period?", "session_id": "demo-1"}
RESUME = {"job": {"job_id": "JOB-1", "job_title": "Data Analyst", "required_skills": "Python, SQL, Excel", "nice_to_have_skills": "Tableau",
                  "min_experience_years": 2, "description_text": "Analyse HR data and build dashboards for business teams."},
          "resumes": [{"resume_id": "R-1", "resume_text": "Data analyst with 4 years of experience using Python, SQL and Tableau to build reports."},
                      {"resume_id": "R-2", "resume_text": "Retail supervisor with 6 years of experience in team management and customer service."}]}
PAYBAND = {"job_level": 3, "prior_experience_years": 5, "market_index": 1.2, "role_id": "R001", "department_id": "D01",
           "work_location": "Bengaluru", "highest_education": "Bachelors"}
FORECAST = {"department_id": "ALL", "horizon_months": 6}
