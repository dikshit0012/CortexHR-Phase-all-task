import pytest

pytestmark = pytest.mark.real_model


def test_health_reports_real_model_status(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    h = r.json()
    assert h["application"] == "up" and h["api_version"]
    assert set(h["models"]) == {"attrition", "attendance_anomaly", "payroll_anomaly", "chatbot", "resume_ranking", "payband", "headcount_forecast"}
    # availability lists must be consistent with per-model status (no false "all available")
    for name, m in h["models"].items():
        assert (name in h["models_available"]) == (m["status"] == "loaded")
        assert (name in h["models_unavailable"]) == (m["status"] != "loaded")
    assert h["all_models_loaded"] == (not h["models_unavailable"])


def test_health_has_no_paths_or_secrets(client):
    text = client.get("/v1/health").text
    assert "/home/" not in text and "test-key" not in text and "\\Users\\" not in text


def test_ready_matches_core_models(client):
    r = client.get("/v1/ready")
    body = r.json()
    assert r.status_code == (200 if body["ready"] else 503)
    for n in ("attrition", "chatbot", "resume_ranking", "payband", "headcount_forecast"):
        if n in body["blocking"]:
            assert not client.get("/v1/health").json()["models"][n]["status"] == "loaded"


def test_root_links(client):
    j = client.get("/").json()
    assert j["links"]["documentation"] == "/docs" and j["links"]["health"] == "/v1/health"
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
