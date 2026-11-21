import json

import pytest

from tests.conftest import make_client
from tests.samples import PAYBAND

pytestmark = pytest.mark.real_model


def test_midpoint_from_selected_model_and_no_invented_band(client, headers):
    r = client.post("/v1/payband/analyze", json=PAYBAND, headers=headers)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["selected_model"] == "gradient_boosting" and b["model_version"] == "paybands-v1"
    assert b["predicted_salary_mid"] > 0 and set(b["comparison"]) == {"linear_regression", "average_by_role_baseline"}
    if "calibration file not configured" in b["band_status"]:
        assert b["recommended_salary_low"] is None and b["recommended_salary_high"] is None
    assert "gender" in b["fairness_note"].lower() and "decision support" in b["decision_note"].lower()


def test_more_experience_changes_prediction(client, headers):
    a = client.post("/v1/payband/analyze", json=PAYBAND, headers=headers).json()["predicted_salary_mid"]
    b = client.post("/v1/payband/analyze", json=dict(PAYBAND, job_level=6), headers=headers).json()["predicted_salary_mid"]
    assert a != b


def test_calibration_file_gives_ordered_band(tmp_path, headers):
    cal = tmp_path / "cal.json"                                      # test configuration only: ratios are supplied, not invented by the API
    cal.write_text(json.dumps({"global": {"low_ratio": 0.9, "high_ratio": 1.1}}))
    with make_client(payband_calibration=cal) as c:
        b = c.post("/v1/payband/analyze", json=PAYBAND, headers=headers).json()
    assert b["recommended_salary_low"] <= b["predicted_salary_mid"] <= b["recommended_salary_high"] and "calibrated" in b["band_status"]


def test_protected_attributes_rejected(client, headers):
    for extra in ({"gender": "F"}, {"age_years": 30}, {"career_gap_months": 3}):
        assert client.post("/v1/payband/analyze", json=dict(PAYBAND, **extra), headers=headers).status_code == 422
    assert client.post("/v1/payband/analyze", json=dict(PAYBAND, job_level=99), headers=headers).status_code == 422
    assert client.post("/v1/payband/analyze", json=PAYBAND).status_code == 401
