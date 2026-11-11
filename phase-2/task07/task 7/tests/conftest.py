"""Shared SYNTHETIC fixtures for Task 7 tests. No real payroll or attendance data is used here."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cortexhr_task7  # noqa: E402,F401  (puts Task 5 / Task 6 on the path)

from cortexhr_task6.detection import combine_flags  # noqa: E402
from cortexhr_task6.payroll_features import build_payroll_features  # noqa: E402
from cortexhr_task6.payroll_rules import apply_rules  # noqa: E402


def payslip(pid, emp="E1", month="2025-04-01", role="R1", fixed=50000.0, ot_hours=0.0, ot_eligible="N",
            variable=0.0, payable_days=None, **over):
    """One internally consistent payslip (amounts in INR); any column can be overridden afterwards."""
    m = pd.Timestamp(month)
    pd_days = float(m.days_in_month) if payable_days is None else payable_days
    ot_pay = 2 * fixed / 208 * ot_hours
    gross = fixed * pd_days / m.days_in_month + ot_pay + variable
    row = dict(payroll_item_id=pid, run_id=f"PR-{m:%Y-%m}", employee_id=emp, month=m, department_id="D01",
               role_id=role, job_level=2, overtime_eligible=ot_eligible, monthly_fixed_gross=fixed,
               payable_days=pd_days, lop_days=m.days_in_month - pd_days, overtime_hours_paid=ot_hours,
               overtime_pay=ot_pay, variable_pay=variable, variable_pay_type=None, gross_pay=gross,
               pf_employee=1800.0, esi_employee=0.0, professional_tax=200.0, tds=3000.0, other_deductions=0.0)
    row["total_deductions"] = (row["pf_employee"] + row["esi_employee"] + row["professional_tax"]
                               + row["tds"] + row["other_deductions"])
    row["net_pay"] = gross - row["total_deductions"]
    row.update(over)
    return row


def attendance_row(aid, emp="E1", month="2025-04-01", scheduled=22.0, present=21.0, paid_leave=1.0, streak=0.0,
                   hours=8.5, **over):
    row = dict(attendance_feature_id=aid, employee_id=emp, month=pd.Timestamp(month), department_id="D01", job_level=2,
               scheduled_days=scheduled, present_days=present, wfh_days=2.0, half_days_unpaid=0.0,
               paid_leave_days=paid_leave, unpaid_absence_days=0.0, longest_unpaid_streak_days=streak,
               late_arrivals=1.0, avg_hours_worked=hours, overtime_hours=0.0,
               attendance_rate=present / scheduled if scheduled else 0.0)
    row.update(over)
    return row


def stub_flags(feats: pd.DataFrame, model_flags=None, rule_cfg=None) -> pd.DataFrame:
    """Task 6 flags with the REAL rules but a hand-set model flag, so unit tests do not depend on the forest.
    The score is a function of the row's CONTENT (gross pay), never of its position."""
    rules = apply_rules(feats, rule_cfg)
    mf = pd.Series(0 if model_flags is None else model_flags, index=feats.index)
    gross = feats["gross_pay"].fillna(0)
    score = (gross / gross.max()) if gross.max() > 0 else gross * 0.0
    model_out = pd.DataFrame({"if_flag": mf.astype(int), "if_anomaly_score": score}, index=feats.index)
    return combine_flags(model_out, rules)


@pytest.fixture
def clean_panel():
    """30 employees x 6 months of clean payslips: pay differs between employees but is CONSTANT per employee, 4 roles."""
    rng = np.random.default_rng(0)
    fixed_by_emp = {e: float(rng.integers(40, 80) * 1000) for e in range(30)}
    rows, n = [], 0
    for e in range(30):
        for m in range(1, 7):
            n += 1
            rows.append(payslip(f"P{n:04d}", emp=f"E{e:02d}", month=f"2025-{m:02d}-01", role=f"R{e % 4}", fixed=fixed_by_emp[e]))
    return pd.DataFrame(rows)


@pytest.fixture
def planted_panel(clean_panel):
    """Clean panel plus planted problems; returns (frame, {name: payroll_item_id})."""
    df = clean_panel.copy()
    ids = {}
    i = df.index[(df.employee_id == "E01") & (df.month == "2025-06-01")][0]          # negative net
    df.loc[i, ["tds", "total_deductions", "net_pay"]] = [90000.0, df.loc[i, "total_deductions"] + 87000.0,
                                                          df.loc[i, "gross_pay"] - df.loc[i, "total_deductions"] - 87000.0]
    ids["negative_net"] = df.loc[i, "payroll_item_id"]
    i = df.index[(df.employee_id == "E02") & (df.month == "2025-06-01")][0]          # wrong decimal (gross x10)
    df.loc[i, "gross_pay"] *= 10
    df.loc[i, "net_pay"] = df.loc[i, "gross_pay"] - df.loc[i, "total_deductions"]
    ids["wrong_decimal"] = df.loc[i, "payroll_item_id"]
    i = df.index[(df.employee_id == "E03") & (df.month == "2025-06-01")][0]          # zero pay
    df.loc[i, ["gross_pay", "net_pay", "total_deductions"]] = 0.0
    ids["zero_pay"] = df.loc[i, "payroll_item_id"]
    i = df.index[(df.employee_id == "E04") & (df.month == "2025-06-01")][0]          # overtime above cap
    df.loc[i, ["overtime_eligible", "overtime_hours_paid"]] = ["Y", 60.0]
    df.loc[i, "overtime_pay"] = 2 * df.loc[i, "monthly_fixed_gross"] / 208 * 60
    df.loc[i, "gross_pay"] += df.loc[i, "overtime_pay"]
    df.loc[i, "net_pay"] = df.loc[i, "gross_pay"] - df.loc[i, "total_deductions"]
    ids["overtime"] = df.loc[i, "payroll_item_id"]
    i = df.index[(df.employee_id == "E05") & (df.month == "2025-06-01")][0]          # PF stopped
    df.loc[i, ["pf_employee"]] = 0.0
    df.loc[i, "total_deductions"] -= 1800.0
    df.loc[i, "net_pay"] += 1800.0
    ids["missed_pf"] = df.loc[i, "payroll_item_id"]
    dup = df.loc[df.index[(df.employee_id == "E06") & (df.month == "2025-05-01")][0]].copy()   # duplicate payslip
    ids["duplicate_original"] = dup["payroll_item_id"]
    dup["payroll_item_id"] = "P9999"
    ids["duplicate_copy"] = "P9999"
    df = pd.concat([df, dup.to_frame().T.astype(df.dtypes.to_dict())], ignore_index=True)
    return df, ids


@pytest.fixture
def clean_attendance():
    rng = np.random.default_rng(1)
    rows, n = [], 0
    for e in range(30):
        for m in range(1, 7):
            n += 1
            rows.append(attendance_row(f"A{n:04d}", emp=f"E{e:02d}", month=f"2025-{m:02d}-01",
                                       present=float(rng.integers(19, 22)), hours=float(rng.choice([8.0, 8.5, 9.0]))))
    return pd.DataFrame(rows)


@pytest.fixture
def planted_attendance(clean_attendance):
    df = clean_attendance.copy()
    ids = {}
    i = df.index[(df.employee_id == "E01") & (df.month == "2025-06-01")][0]
    df.loc[i, ["present_days", "paid_leave_days", "attendance_rate"]] = [0.0, 0.0, 0.0]; ids["no_attendance"] = df.loc[i, "attendance_feature_id"]
    i = df.index[(df.employee_id == "E02") & (df.month == "2025-06-01")][0]
    df.loc[i, "longest_unpaid_streak_days"] = 8.0; ids["long_unpaid"] = df.loc[i, "attendance_feature_id"]
    i = df.index[(df.employee_id == "E03") & (df.month == "2025-06-01")][0]
    df.loc[i, "avg_hours_worked"] = 16.0; ids["hours"] = df.loc[i, "attendance_feature_id"]
    return df, ids
