"""Attendance and payroll anomaly endpoints.

Real inference needs the private reference workbook AND the exact library versions the artifacts were built with (the Task 8
loader refuses anything else). Without them the endpoints must answer 503 - never a guess. Both behaviours are tested.
"""
import pytest

from tests.conftest import WORKBOOK
from tests.samples import ATTENDANCE, PAYROLL


def _is_loaded(client, name):
    return client.get("/v1/health").json()["models"][name]["status"] == "loaded"


@pytest.mark.mocked
def test_503_not_a_guess_when_detectors_unavailable(client, headers):
    for path, name, rec in (("attendance", "attendance_anomaly", ATTENDANCE), ("payroll", "payroll_anomaly", PAYROLL)):
        if _is_loaded(client, name):
            pytest.skip("detectors are loaded in this environment")
        r = client.post(f"/v1/anomaly/{path}", json={"records": [rec]}, headers=headers)
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"
        assert "results" not in r.json()


@pytest.mark.mocked
def test_schema_validation_runs_before_model_availability(client, headers):
    bad = dict(ATTENDANCE, attendance_rate=3)
    assert client.post("/v1/anomaly/attendance", json={"records": [bad]}, headers=headers).status_code == 422
    bad_p = dict(PAYROLL, run_id="PR-2020-01")
    assert client.post("/v1/anomaly/payroll", json={"records": [bad_p]}, headers=headers).status_code == 422
    assert client.post("/v1/anomaly/payroll", json={"records": []}, headers=headers).status_code == 422
    assert client.post("/v1/anomaly/payroll", json={"records": [PAYROLL]}).status_code == 401


@pytest.mark.real_model
@pytest.mark.needs_workbook
@pytest.mark.skipif(not WORKBOOK, reason="needs CORTEXHR_WORKBOOK (private reference workbook)")
def test_real_inference(client, headers):
    if not (_is_loaded(client, "attendance_anomaly") and _is_loaded(client, "payroll_anomaly")):
        pytest.skip("detectors did not load (workbook / library versions) - see /v1/health detail")
    a = client.post("/v1/anomaly/attendance", json={"records": [ATTENDANCE]}, headers=headers)
    assert a.status_code == 200, a.text
    r = a.json()["results"][0]
    assert 0 <= r["anomaly_score"] <= 1 and r["flagged"] in ("Y", "N") and r["model_version"]
    p = client.post("/v1/anomaly/payroll", json={"records": [dict(PAYROLL, net_pay=-500.0)]}, headers=headers)
    assert p.status_code == 200, p.text
