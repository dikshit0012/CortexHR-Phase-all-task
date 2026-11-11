"""Shared synthetic fixtures - no real payroll data is used in any test."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def payslip(pid, emp="E1", month="2025-04-01", role="R1", fixed=50000.0, ot_hours=0.0, ot_eligible="N",
            variable=0.0, payable_days=None, **over):
    """One internally consistent payslip; any column can be overridden afterwards."""
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


@pytest.fixture
def make_frame():
    def _make(rows):
        return pd.DataFrame(rows)
    return _make


@pytest.fixture
def base_frame():
    """3 employees x 4 months of clean payslips (employee E3 is on a different role)."""
    rows, n = [], 0
    for emp, role, fixed in [("E1", "R1", 50000.0), ("E2", "R1", 52000.0), ("E3", "R2", 90000.0)]:
        for mo in ["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01"]:
            n += 1
            rows.append(payslip(f"P{n:03d}", emp=emp, month=mo, role=role, fixed=fixed))
    return pd.DataFrame(rows)
