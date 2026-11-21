import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from tests.samples import ATTRITION

pytestmark = pytest.mark.real_model
ART = Path(__file__).resolve().parents[1] / "artifacts"


def _direct_probability(rows):
    meta = json.loads((ART / "attrition-v2.metadata.json").read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(ART / "attrition-v2.joblib")
    X = pd.DataFrame([{k: v for k, v in r.items() if k != "record_ref"} for r in rows])[meta["feature_list"]]
    return model.predict_proba(X)[:, 1]


def test_predict_matches_saved_model_exactly(client, headers):
    r = client.post("/v1/attrition/predict", json={"records": [ATTRITION]}, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_version"] == "attrition-v2" and body["n_records"] == 1
    p = body["results"][0]["attrition_probability_12m"]
    assert 0 <= p <= 1
    assert p == pytest.approx(float(_direct_probability([ATTRITION])[0]), abs=1e-12)   # API == artifact, no substitute
    assert body["results"][0]["record_ref"] == "demo-1" and body["results"][0]["risk_band"] is None


def test_probability_responds_to_inputs(client, headers):
    low = dict(ATTRITION, last_engagement_score=5, comp_ratio=1.3, attendance_rate_3m=0.99)
    high = dict(ATTRITION, last_engagement_score=1, comp_ratio=0.7, attendance_rate_3m=0.7)
    p = [x["attrition_probability_12m"] for x in client.post("/v1/attrition/predict", json={"records": [low, high]}, headers=headers).json()["results"]]
    assert p[0] != p[1]


def test_batch_of_ten_gets_official_risk_bands(client, headers):
    recs = [dict(ATTRITION, record_ref=f"r{i}", last_engagement_score=1 + i * 0.4, tenure_months=6 + i * 9) for i in range(10)]
    res = client.post("/v1/attrition/predict", json={"records": recs}, headers=headers).json()["results"]
    bands = [x["risk_band"] for x in res]
    assert set(bands) <= {"Low", "Medium", "High"} and "High" in bands and "Low" in bands
    top = max(res, key=lambda x: x["attrition_probability_12m"])
    assert top["risk_band"] == "High"


def test_missing_values_are_imputed(client, headers):
    rec = {k: v for k, v in ATTRITION.items() if k not in ("last_engagement_score", "work_mode")}
    assert client.post("/v1/attrition/predict", json={"records": [rec]}, headers=headers).status_code == 200


def test_invalid_input_rejected_without_echoing_values(client, headers):
    r = client.post("/v1/attrition/predict", json={"records": [dict(ATTRITION, attendance_rate_3m=7.5)]}, headers=headers)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert "7.5" not in r.text


def test_unknown_and_forbidden_fields_rejected(client, headers):
    for extra in ({"left_within_12m": 1}, {"employee_name": "X"}, {"gender": "F"}):
        assert client.post("/v1/attrition/predict", json={"records": [dict(ATTRITION, **extra)]}, headers=headers).status_code == 422


def test_empty_and_oversized_batches(client, headers):
    assert client.post("/v1/attrition/predict", json={"records": []}, headers=headers).status_code == 422
    big = {"records": [ATTRITION] * 101}
    assert client.post("/v1/attrition/predict", json=big, headers=headers).status_code == 400


def test_requires_api_key(client):
    assert client.post("/v1/attrition/predict", json={"records": [ATTRITION]}).status_code == 401
    assert client.post("/v1/attrition/predict", json={"records": [ATTRITION]}, headers={"X-API-Key": "wrong"}).status_code == 401
