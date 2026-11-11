import pytest

from helpers import PRIORITY, assert_official_row

SRC = "attendance_monthly_features"


def no_show(make_attendance, aid="AF-NEW-NS"):
    return make_attendance(aid, present_days=0, paid_leave_days=0, unpaid_absence_days=21, longest_unpaid_streak_days=21,
                           late_arrivals=0, avg_hours_worked=0.0, overtime_hours=0.0, attendance_rate=0.0)


# ---- Test 1: valid attendance request -> 200, official format --------------------------------------------------
def test_valid_attendance_request_returns_official_format(client, make_attendance):
    r = client.post("/v1/anomaly/attendance", json=no_show(make_attendance))
    assert r.status_code == 200
    row = r.json()
    assert_official_row(row, SRC, "AF-NEW-NS", flagged="Y")
    assert row["employee_id"] == "EMP-0001" and row["period"] == "2026-07-01"
    assert row["result_id"] == "AN-0001"
    # Task 7 rule for attendance: no euro/rupee amount can be supported, so money at risk stays empty (null)
    assert row["money_at_risk"] is None


def test_normal_attendance_record_is_scored_and_follows_the_same_schema(client, make_attendance):
    r = client.post("/v1/anomaly/attendance", json=make_attendance())
    assert r.status_code == 200
    assert_official_row(r.json(), SRC, "AF-NEW-01")


# ---- Test 3: invalid attendance request -> 422 -------------------------------------------------------------------
def test_incomplete_attendance_request_is_422(client):
    r = client.post("/v1/anomaly/attendance", json={"employee_id": "EMP-0001"})
    assert r.status_code == 422
    missing = {tuple(e["loc"]) for e in r.json()["detail"] if e["type"] == "missing"}
    assert ("body", "attendance_feature_id") in missing and ("body", "scheduled_days") in missing


@pytest.mark.parametrize("override", [
    {"present_days": -1},                         # negative count
    {"present_days": 20.5},                       # not a whole number of days
    {"attendance_rate": 1.5},                     # a rate cannot exceed 1
    {"avg_hours_worked": "abc"},                  # not a number
    {"month": "07/2026"},                         # not a date
    {"employee_id": ""},                          # blank id
    {"employee_id": None},                        # id may not be null
    {"surprise_field": 1},                        # unknown field is rejected
])
def test_invalid_attendance_values_are_422(client, make_attendance, override):
    assert client.post("/v1/anomaly/attendance", json=make_attendance(**override)).status_code == 422


def test_non_finite_number_is_422(client, make_attendance):
    body = '{"attendance_feature_id":"A","employee_id":"E","month":"2026-07-01","scheduled_days":21,"present_days":20,' \
           '"half_days_unpaid":0,"paid_leave_days":1,"unpaid_absence_days":0,"longest_unpaid_streak_days":0,' \
           '"late_arrivals":1,"avg_hours_worked":Infinity,"overtime_hours":0,"attendance_rate":0.9}'
    r = client.post("/v1/anomaly/attendance", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_malformed_json_is_422(client):
    r = client.post("/v1/anomaly/attendance", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_null_measure_is_scored_not_rejected(client, make_attendance):
    """Task 5 imputes missing measures, so a null measure is a valid (scoreable) request."""
    r = client.post("/v1/anomaly/attendance", json=make_attendance(avg_hours_worked=None))
    assert r.status_code == 200


# ---- multiple flags on one record ---------------------------------------------------------------------------------
def test_record_with_several_rule_flags_reports_the_highest_priority_reason(client, make_attendance):
    """No presence AND a 21-day unpaid streak both fire; the response carries ONE reason: the Task 7 priority winner."""
    row = client.post("/v1/anomaly/attendance", json=no_show(make_attendance)).json()
    competing = {"NO_ATTENDANCE", "LONG_UNPAID_ABSENCE"}
    assert row["reason_code"] in competing
    assert PRIORITY[row["reason_code"]] == min(PRIORITY[c] for c in competing)
