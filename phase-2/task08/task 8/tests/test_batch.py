import pytest

from conftest import TEST_BATCH_LIMIT, gross_above_norm_payroll
from helpers import assert_official_row


def attendance_batch(make_attendance, n):
    return [make_attendance(f"AF-B-{i:02d}", f"EMP-{i + 1:04d}") for i in range(n)]


def test_batch_with_multiple_valid_attendance_rows(client, make_attendance):
    recs = attendance_batch(make_attendance, 4)
    r = client.post("/v1/anomaly/attendance/batch", json={"records": recs})
    assert r.status_code == 200
    body = r.json()
    assert body["detector"] == "attendance" and body["count"] == 4 and len(body["results"]) == 4
    for rec, row in zip(recs, body["results"]):
        assert_official_row(row, "attendance_monthly_features", rec["attendance_feature_id"])


def test_batch_results_keep_input_order(client, make_attendance):
    recs = attendance_batch(make_attendance, 5)[::-1]                    # deliberately not sorted
    body = client.post("/v1/anomaly/attendance/batch", json={"records": recs}).json()
    assert [x["record_id"] for x in body["results"]] == [x["attendance_feature_id"] for x in recs]


def test_batch_payroll_ranks_by_money_and_result_ids_are_unique(client, make_payroll):
    small = gross_above_norm_payroll("PI-B-SMALL", "EMP-0002")
    small.update(gross_pay=200000.0, tds=20000.0, total_deductions=22000.0, net_pay=178000.0)
    big = gross_above_norm_payroll("PI-B-BIG", "EMP-0003")
    ok = make_payroll("PI-B-OK", "EMP-0004")
    body = client.post("/v1/anomaly/payroll/batch", json={"records": [ok, small, big]}).json()
    assert [x["record_id"] for x in body["results"]] == ["PI-B-OK", "PI-B-SMALL", "PI-B-BIG"]   # input order
    ids = [x["result_id"] for x in body["results"]]
    assert len(set(ids)) == 3
    flagged = sorted((x for x in body["results"] if x["flagged"] == "Y"), key=lambda x: x["result_id"])
    money = [x["money_at_risk"] for x in flagged]
    assert money == sorted(money, reverse=True)                    # AN-0001 carries the most money at risk
    assert flagged[0]["record_id"] == "PI-B-BIG"
    assert all(x["result_id"] > flagged[-1]["result_id"] for x in body["results"] if x["flagged"] == "N")  # N after Y


def test_score_does_not_depend_on_the_other_records_in_the_batch(client, make_payroll):
    """Fixed reference scaling: the same record gets the same score and reason alone or inside a batch."""
    target = gross_above_norm_payroll("PI-S-1", "EMP-0005")
    alone = client.post("/v1/anomaly/payroll", json=target).json()
    others = [make_payroll(f"PI-S-O{i}", f"EMP-{i + 6:04d}") for i in range(3)]
    batch = client.post("/v1/anomaly/payroll/batch", json={"records": others + [target]}).json()["results"][-1]
    for col in ("anomaly_score", "reason_code", "reason_text", "money_at_risk", "flagged"):
        assert batch[col] == alone[col]


def test_attendance_score_does_not_depend_on_batch_composition(client, make_attendance):
    target = make_attendance("AF-S-1", "EMP-0005")
    alone = client.post("/v1/anomaly/attendance", json=target).json()
    others = [make_attendance(f"AF-S-O{i}", f"EMP-{i + 6:04d}") for i in range(3)]
    batch = client.post("/v1/anomaly/attendance/batch", json={"records": others + [target]}).json()["results"][-1]
    for col in ("anomaly_score", "reason_code", "reason_text", "money_at_risk", "flagged"):
        assert batch[col] == alone[col]


def test_empty_batch_is_rejected(client):
    for path in ("attendance", "payroll"):
        assert client.post(f"/v1/anomaly/{path}/batch", json={"records": []}).status_code == 422


def test_oversized_batch_is_rejected_with_the_limit_in_the_message(client, make_attendance):
    recs = attendance_batch(make_attendance, TEST_BATCH_LIMIT + 1)
    r = client.post("/v1/anomaly/attendance/batch", json={"records": recs})
    assert r.status_code == 422
    assert str(TEST_BATCH_LIMIT) in r.json()["detail"][0]["msg"]
    assert client.post("/v1/anomaly/attendance/batch", json={"records": recs[:TEST_BATCH_LIMIT]}).status_code == 200


def test_one_invalid_record_rejects_the_whole_batch(client, make_attendance):
    recs = attendance_batch(make_attendance, 3)
    recs[1]["present_days"] = -4
    r = client.post("/v1/anomaly/attendance/batch", json={"records": recs})
    assert r.status_code == 422                                     # all-or-nothing: no partial results
    assert "results" not in r.json()
    assert ["body", "records", 1, "present_days"] in [e["loc"] for e in r.json()["detail"]]


def test_duplicate_ids_inside_a_batch_are_rejected(client, make_attendance):
    rec = make_attendance()
    assert client.post("/v1/anomaly/attendance/batch", json={"records": [rec, rec]}).status_code == 422


def test_unknown_detector_name_is_404(client, make_attendance):
    assert client.post("/v1/anomaly/turnover/batch", json={"records": [make_attendance()]}).status_code == 404
    assert client.post("/v1/anomaly/turnover", json=make_attendance()).status_code == 404
