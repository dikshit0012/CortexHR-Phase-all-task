import pytest

from helpers import PRIORITY, assert_official_row
from conftest import gross_above_norm_payroll

SRC = "payroll_monthly_features"


# ---- Test 2: valid payroll request -> 200, official format -------------------------------------------------------
def test_valid_payroll_request_returns_official_format(client):
    r = client.post("/v1/anomaly/payroll", json=gross_above_norm_payroll())
    assert r.status_code == 200
    row = r.json()
    assert_official_row(row, SRC, "PI-NEW-HI", flagged="Y")
    assert row["employee_id"] == "EMP-0001" and row["period"] == "2026-07-01"
    assert row["reason_code"] == "GROSS_ABOVE_NORM"


def test_normal_payslip_is_scored_and_follows_the_same_schema(client, make_payroll):
    r = client.post("/v1/anomaly/payroll", json=make_payroll())
    assert r.status_code == 200
    assert_official_row(r.json(), SRC, "PI-NEW-01")


# ---- Test 6 (part): money at risk follows the Task 7 formula -----------------------------------------------------
def test_gross_above_norm_money_is_gross_minus_expected_gross(client):
    row = client.post("/v1/anomaly/payroll", json=gross_above_norm_payroll()).json()
    expected_gross = 46500.0 + 0.0 + 4000.0                    # fixed x payable/days_in_month + overtime + variable
    assert row["money_at_risk"] == pytest.approx(465000.0 - expected_gross, abs=0.01)


def test_duplicate_payslip_money_is_not_counted_twice(client, frames, make_payroll):
    """A new payslip that repeats an existing payslip: Task 7 splits (copies-1) x net over the flagged copies,
    and the copy already in the reference history is one of them, so this record carries HALF the net pay."""
    existing = frames[1].iloc[-1].to_dict()                    # EMP-0040, 2026-06
    dup = make_payroll("PI-NEW-DUP", existing["employee_id"], role_id=existing["role_id"], month="2026-06-01",
                       run_id="PR-2026-06", **{k: existing[k] for k in (
                           "monthly_fixed_gross", "payable_days", "lop_days", "overtime_hours_paid", "overtime_pay",
                           "variable_pay", "gross_pay", "pf_employee", "esi_employee", "professional_tax", "tds",
                           "other_deductions", "total_deductions", "net_pay", "overtime_eligible", "job_level")})
    row = client.post("/v1/anomaly/payroll", json=dup).json()
    assert row["reason_code"] == "DUPLICATE_PAYMENT"
    assert row["money_at_risk"] == pytest.approx(existing["net_pay"] / 2, abs=0.01)


def test_negative_net_with_other_flags_reports_one_reason_and_one_amount(client, make_payroll):
    """Negative net pay AND a gross that is far too high: one reason (priority winner), one amount (not summed)."""
    rec = make_payroll("PI-NEW-NEG", gross_pay=465000.0, tds=0.0, total_deductions=2000.0, net_pay=-5000.0)
    row = client.post("/v1/anomaly/payroll", json=rec).json()
    assert row["flagged"] == "Y"
    competing = {"NEGATIVE_NET_PAY", "GROSS_ABOVE_NORM"}
    assert row["reason_code"] in competing
    assert PRIORITY[row["reason_code"]] == min(PRIORITY[c] for c in competing)
    gross_amount = 465000.0 - 50500.0
    assert row["money_at_risk"] < gross_amount + 5000.0       # the two issues are not added together


# ---- Test 4: invalid payroll request -> 422 ---------------------------------------------------------------------
def test_incomplete_payroll_request_is_422(client):
    r = client.post("/v1/anomaly/payroll", json={"employee_id": "EMP-0001", "net_pay": 1.0})
    assert r.status_code == 422
    missing = {tuple(e["loc"]) for e in r.json()["detail"] if e["type"] == "missing"}
    assert ("body", "payroll_item_id") in missing and ("body", "gross_pay") in missing


@pytest.mark.parametrize("override", [
    {"overtime_eligible": "maybe"},               # only Y / N exist
    {"month": "2026-07-15"},                      # Task 6 needs the first day of the month
    {"month": "July"},                            # not a date
    {"run_id": "PR-2026-06"},                     # run_id must match the payroll month
    {"gross_pay": "lots"},                        # not a number
    {"payable_days": -1},                         # negative days
    {"net_pay": None, "gross_pay": "x"},          # a bad value is still rejected next to a null
    {"payroll_item_id": "   "},                   # blank id
    {"extra_column": 5},                          # unknown field
])
def test_invalid_payroll_values_are_422(client, make_payroll, override):
    assert client.post("/v1/anomaly/payroll", json=make_payroll(**override)).status_code == 422


def test_missing_amount_is_scored_as_invalid_payroll_data_not_rejected(client, make_payroll):
    """Task 6 has an explicit rule for a missing amount, so null is valid input and the rule explains it."""
    row = client.post("/v1/anomaly/payroll", json=make_payroll("PI-NEW-NULL", gross_pay=None)).json()
    assert row["flagged"] == "Y" and row["reason_code"] == "INVALID_PAYROLL_DATA"
    assert row["money_at_risk"] is None


def test_negative_net_pay_is_accepted_as_input(client, make_payroll):
    assert client.post("/v1/anomaly/payroll", json=make_payroll("PI-NEW-N2", net_pay=-100.0)).status_code == 200
