import json
from pathlib import Path

import pytest
from fastapi import APIRouter

from cortexhr_api.main import create_app
from cortexhr_api.config import Settings
from cortexhr_api.registry import ModelRegistry
from tests.conftest import make_client
from tests.samples import ATTRITION, CHAT

pytestmark = pytest.mark.mocked


def test_auth_fails_closed_when_no_key_configured():
    with make_client(api_keys=()) as c:
        r = c.post("/v1/chat", json=CHAT, headers={"X-API-Key": "anything"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "auth_not_configured"
        ready = c.get("/v1/ready")
        assert ready.status_code == 503 and "authentication" in ready.json()["blocking"]


def test_auth_can_be_disabled_explicitly_for_local_dev():
    with make_client(require_auth=False, api_keys=()) as c:
        assert c.post("/v1/chat", json=CHAT).status_code == 200


def test_rate_limit_returns_429_with_retry_after():
    with make_client(rate_limit_per_minute=3) as c:
        codes = [c.post("/v1/chat", json=CHAT, headers={"X-API-Key": "test-key"}).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
        r = c.post("/v1/chat", json=CHAT, headers={"X-API-Key": "test-key"})
        assert "retry-after" in {k.lower() for k in r.headers}
        assert c.get("/v1/health").status_code == 200                 # health is not rate limited


def test_request_body_size_limit_413():
    with make_client(max_body_bytes=2000) as c:
        r = c.post("/v1/attrition/predict", json={"records": [ATTRITION] * 20}, headers={"X-API-Key": "test-key"})
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"


def test_cors_only_for_configured_origins():
    with make_client(cors_origins=("https://hr.example.com",)) as c:
        ok = c.options("/v1/chat", headers={"Origin": "https://hr.example.com", "Access-Control-Request-Method": "POST"})
        bad = c.options("/v1/chat", headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"})
        assert ok.headers.get("access-control-allow-origin") == "https://hr.example.com"
        assert "access-control-allow-origin" not in bad.headers


def test_internal_errors_never_leak_details():
    app = create_app(Settings(api_keys=("k",)))
    boom = APIRouter()

    @boom.get("/boom")
    def _boom():
        raise RuntimeError("secret /etc/passwd salary=123456 traceback")

    app.include_router(boom)
    from fastapi.testclient import TestClient
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom")
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
    assert "passwd" not in r.text and "salary" not in r.text and "Traceback" not in r.text and r.headers["x-request-id"]


def test_security_headers_and_request_id():
    with make_client() as c:
        r = c.get("/v1/health", headers={"X-Request-ID": "abc-123"})
        assert r.headers["x-request-id"] == "abc-123" and r.headers["x-content-type-options"] == "nosniff"
        assert c.get("/v1/health", headers={"X-Request-ID": "bad id\n!"}).headers["x-request-id"] != "bad id\n!"


def test_unknown_route_is_clean_404(client):
    r = client.get("/v1/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_missing_artifacts_mark_model_unavailable_not_crash(tmp_path, headers):
    with make_client(artifact_dir=tmp_path) as c:                      # empty artifact directory
        h = c.get("/v1/health").json()
        assert "attrition" in h["models_unavailable"] and "chatbot" in h["models_unavailable"] and not h["all_models_loaded"]
        assert "payband" in h["models_unavailable"]
        r = c.post("/v1/attrition/predict", json={"records": [ATTRITION]}, headers={"X-API-Key": "test-key"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable" and str(tmp_path) not in r.text
        assert c.get("/v1/ready").status_code == 503
        assert c.get("/v1/health").status_code == 200                  # application stays healthy while models are missing


def test_logs_contain_no_request_bodies(client, headers, capsys):
    client.post("/v1/attrition/predict", json={"records": [dict(ATTRITION, record_ref="PRIVATE-REF-777")]}, headers=headers)
    out = capsys.readouterr().out
    assert "PRIVATE-REF-777" not in out and "900000" not in out


def test_workbook_download_requires_https_and_matching_hash(tmp_path):
    import hashlib
    from cortexhr_api.registry import fetch_workbook
    dest = tmp_path / "wb.xlsx"
    with pytest.raises(ValueError, match="https"):
        fetch_workbook(Settings(workbook_path=dest, workbook_url="http://example.com/x.xlsx", workbook_sha256="0" * 64))
    with pytest.raises(ValueError, match="SHA256 is required"):
        fetch_workbook(Settings(workbook_path=dest, workbook_url="https://example.com/x.xlsx"))
    assert not dest.exists()
