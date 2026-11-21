"""End-to-end flow across all endpoints with the real artifacts (anomaly detectors only when their private data is provided)."""
import pytest

from tests.samples import ATTRITION, CHAT, FORECAST, PAYBAND, RESUME

pytestmark = pytest.mark.real_model


def test_every_core_endpoint_in_one_session(client, headers):
    assert client.get("/v1/ready").status_code == 200
    assert client.post("/v1/attrition/predict", json={"records": [ATTRITION]}, headers=headers).status_code == 200
    assert client.post("/v1/chat", json=CHAT, headers=headers).status_code == 200
    assert client.post("/v1/resume/rank", json=RESUME, headers=headers).status_code == 200
    assert client.post("/v1/payband/analyze", json=PAYBAND, headers=headers).status_code == 200
    assert client.post("/v1/forecast/headcount", json=FORECAST, headers=headers).status_code == 200


def test_models_are_loaded_once_not_per_request(client, headers):
    reg = client.app.state.registry
    ids = {n: id(o) for n, o in reg.objects.items()}
    for _ in range(3):
        client.post("/v1/attrition/predict", json={"records": [ATTRITION]}, headers=headers)
        client.post("/v1/chat", json=CHAT, headers=headers)
    assert ids == {n: id(o) for n, o in client.app.state.registry.objects.items()}


def test_error_shape_is_consistent(client, headers):
    for r in (client.post("/v1/chat", json={}, headers=headers), client.post("/v1/chat", json=CHAT),
              client.get("/v1/missing"), client.post("/v1/forecast/headcount", json={"department_id": "D99"}, headers=headers)):
        e = r.json()["error"]
        assert set(e) == {"code", "message", "request_id"} and e["request_id"]


def test_openapi_documents_every_endpoint_with_tags(client):
    spec = client.get("/openapi.json").json()
    expected = {"/v1/health", "/v1/ready", "/v1/attrition/predict", "/v1/anomaly/attendance", "/v1/anomaly/payroll", "/v1/chat",
                "/v1/resume/rank", "/v1/payband/analyze", "/v1/forecast/headcount"}
    assert expected <= set(spec["paths"])
    assert all(op.get("tags") and op.get("summary") for p in expected for op in spec["paths"][p].values())


def test_frontend_is_served(client):
    r = client.get("/app/")
    assert r.status_code == 200 and "CortexHR" in r.text
    assert client.get("/app/app.js").status_code == 200 and client.get("/app/style.css").status_code == 200
