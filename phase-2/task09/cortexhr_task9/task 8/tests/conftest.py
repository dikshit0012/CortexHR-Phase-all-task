"""SYNTHETIC fixtures only: no real employee data is needed to run these tests.

A small fictional company (40 employees, 6 months) is written to a temporary .xlsx with the two sheets the
service reads, artifacts are built from it with the real build code, and the app is started on those.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import _paths  # noqa: E402,F401
from app.artifacts import build_artifacts  # noqa: E402
from app.bootstrap import load_service  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.reference import ATTENDANCE_SHEET, PAYROLL_SHEET, load_reference  # noqa: E402

MONTHS = pd.date_range("2026-01-01", periods=6, freq="MS")
NEW_MONTH = "2026-07-01"                      # the month the test requests are about (after the history)
TEST_BATCH_LIMIT = 5


def _payslip(pid, emp, month, role, fixed, variable, ot_eligible="N", ot_hours=0.0):
    dim = month.days_in_month
    ot_pay = round(2 * fixed / 208 * ot_hours, 2)
    gross = round(fixed + ot_pay + variable, 2)
    tds = round(0.10 * gross, 2)
    total = round(1800.0 + 0.0 + 200.0 + tds, 2)
    return dict(payroll_item_id=pid, run_id=f"PR-{month:%Y-%m}", employee_id=emp, month=month, department_id="D01",
                role_id=role, job_level=2, overtime_eligible=ot_eligible, monthly_fixed_gross=float(fixed),
                payable_days=float(dim), lop_days=0.0, overtime_hours_paid=float(ot_hours), overtime_pay=ot_pay,
                variable_pay=float(variable), variable_pay_type=None, gross_pay=gross, pf_employee=1800.0,
                esi_employee=0.0, professional_tax=200.0, tds=tds, other_deductions=0.0, total_deductions=total,
                net_pay=round(gross - total, 2))


def make_reference_frames(n_emp=40, seed=7):
    rng = np.random.default_rng(seed)
    pay, att, k = [], [], 0
    for e in range(1, n_emp + 1):
        emp, role = f"EMP-{e:04d}", f"R{e % 5 + 1:03d}"
        fixed = 40000 + 6000 * (e % 5) + 500 * (e % 7)
        for m in MONTHS:
            k += 1
            eligible = "Y" if e % 6 == 0 else "N"
            ot = float(rng.integers(0, 12)) if eligible == "Y" else 0.0
            pay.append(_payslip(f"PI-{k:06d}", emp, m, role, fixed, round(fixed * rng.uniform(0.05, 0.15), 2), eligible, ot))
            sched = int(rng.integers(20, 23))
            leave = int(rng.integers(0, 3))
            att.append(dict(attendance_feature_id=f"AF-{k:05d}", employee_id=emp, month=m, department_id="D01", job_level=2,
                            scheduled_days=sched, present_days=sched - leave, wfh_days=int(rng.integers(0, 8)),
                            half_days_unpaid=0, paid_leave_days=leave, unpaid_absence_days=0,
                            longest_unpaid_streak_days=0, late_arrivals=int(rng.integers(0, 4)),
                            avg_hours_worked=round(float(rng.normal(8.5, 0.3)), 2), overtime_hours=float(rng.integers(0, 7)),
                            attendance_rate=round((sched - leave) / sched, 4)))
    return pd.DataFrame(att), pd.DataFrame(pay)


def write_workbook(path, att, pay, sheets=None):
    with pd.ExcelWriter(path) as xw:
        for name, frame in (sheets or {ATTENDANCE_SHEET: att, PAYROLL_SHEET: pay}).items():
            frame.to_excel(xw, sheet_name=name, index=False)


@pytest.fixture(scope="session")
def frames():
    return make_reference_frames()


@pytest.fixture(scope="session")
def workbook(tmp_path_factory, frames):
    p = tmp_path_factory.mktemp("ref") / "reference.xlsx"
    write_workbook(p, *frames)
    return p


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, workbook):
    d = tmp_path_factory.mktemp("models")
    build_artifacts(load_reference(workbook), d)
    return d


@pytest.fixture(scope="session")
def settings(workbook, model_dir):
    return Settings(workbook_path=workbook, model_dir=model_dir, max_batch_size=TEST_BATCH_LIMIT)


@pytest.fixture(scope="session")
def service(settings):
    return load_service(settings)


@pytest.fixture()
def client(service):
    from fastapi.testclient import TestClient
    with TestClient(create_app(service=service)) as c:
        yield c


# ---- request builders: EMP-0001, new month, consistent with that employee's history -----------------------------
def payroll_request(pid="PI-NEW-01", emp="EMP-0001", **over):
    m = pd.Timestamp(NEW_MONTH)
    row = _payslip(pid, emp, m, "R002", 46500, 4000)           # EMP-0001: fixed = 40000 + 6000*1 + 500*1
    row["month"] = NEW_MONTH
    row.pop("variable_pay_type")
    row.update(over)
    return row


def attendance_request(aid="AF-NEW-01", emp="EMP-0001", **over):
    row = dict(attendance_feature_id=aid, employee_id=emp, month=NEW_MONTH, scheduled_days=21, present_days=20,
               half_days_unpaid=0, paid_leave_days=1, unpaid_absence_days=0, longest_unpaid_streak_days=0,
               late_arrivals=1, avg_hours_worked=8.5, overtime_hours=2.0, attendance_rate=round(20 / 21, 4))
    row.update(over)
    return row


def gross_above_norm_payroll(pid="PI-NEW-HI", emp="EMP-0001"):
    """Gross is 10x what fixed + overtime + variable allow; net / deductions kept consistent with that gross."""
    gross = 465000.0
    tds = round(0.10 * gross, 2)
    total = round(1800 + 200 + tds, 2)
    return payroll_request(pid, emp, gross_pay=gross, tds=tds, total_deductions=total, net_pay=round(gross - total, 2))


@pytest.fixture()
def make_payroll():
    return payroll_request


@pytest.fixture()
def make_attendance():
    return attendance_request
