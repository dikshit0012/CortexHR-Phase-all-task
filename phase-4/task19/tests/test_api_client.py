import logging

import pytest
import requests
import responses

from api_client import ApiClient, ApiError
from config import Config
from tests.conftest import FIX, real

BASE = "https://api.test"


@pytest.fixture
def c():
    return ApiClient(Config(api_url=BASE, timeout=2, api_key="KEY-SECRET-123"))


def err(fn, *a, **k):
    with pytest.raises(ApiError) as e:
        fn(*a, **k)
    return e.value


# ---- success paths with payloads captured from the real API -------------------------------------------------------------
@responses.activate
def test_success_all_routes(c):
    responses.post(f"{BASE}/v1/attrition/predict", json=real("attrition1"))
    responses.post(f"{BASE}/v1/chat", json=real("chat_faq"))
    responses.post(f"{BASE}/v1/resume/rank", json=real("resume"))
    responses.post(f"{BASE}/v1/payband/analyze", json=real("payband"))
    responses.post(f"{BASE}/v1/forecast/headcount", json=real("forecast"))
    assert 0 <= c.attrition_predict([{"job_level": 3}])["results"][0]["attrition_probability_12m"] <= 1
    assert c.chat("What is the notice period?", "s1")["fallback"] is False
    assert c.resume_rank({"job_title": "x"}, [{"resume_id": "C-1", "resume_text": "t"}])["ranking"][0]["rank"] == 1
    assert c.payband({"job_level": 3})["predicted_salary_mid"] > 0
    assert len(c.forecast("ALL", 6)["forecast"]) == 6


@responses.activate
def test_key_header_and_json_body(c):
    responses.post(f"{BASE}/v1/chat", json=real("chat_faq"))
    c.chat("hello", "abc")
    req = responses.calls[0].request
    assert req.headers["X-API-Key"] == "KEY-SECRET-123" and req.headers["Content-Type"] == "application/json"
    assert b"role" not in req.body and b'"session_id": "abc"' in req.body


@responses.activate
def test_no_key_header_when_not_configured():
    responses.get(f"{BASE}/v1/health", json={"application": "up", "models": {}, "models_available": [], "models_unavailable": []})
    ApiClient(Config(api_url=BASE)).health()
    assert "X-API-Key" not in responses.calls[0].request.headers


# ---- real error shapes ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,status,kind", [("no_key", 401, "auth"), ("bad422", 422, "validation"), ("forecast_bad_dept", 404, "not_found"),
                                              ("anomaly_503", 503, "unavailable")])
@responses.activate
def test_real_error_bodies(c, name, status, kind):
    assert FIX[name]["status"] == status
    responses.post(f"{BASE}/v1/chat", json=FIX[name]["body"], status=status)
    e = err(c.chat, "x")
    assert e.kind == kind and e.status == status and e.request_id == FIX[name]["body"]["error"]["request_id"]


@responses.activate
def test_validation_message_helpful_but_no_values(c):
    responses.post(f"{BASE}/v1/attrition/predict", status=422, json={"error": {"code": "validation_error", "request_id": "r1",
                   "message": "Invalid request: records.0.job_level: Input should be less than or equal to 7"}})
    e = err(c.attrition_predict, [{}])
    assert "job_level" in e.user_message and "Invalid request:" not in e.user_message


@responses.activate
def test_503_does_not_leak_server_config_text(c):
    responses.post(f"{BASE}/v1/anomaly/attendance", json=FIX["anomaly_503"]["body"], status=503)
    e = err(c.anomaly, "attendance", [{}])
    assert "CORTEXHR_WORKBOOK" not in e.user_message and "RuntimeError" not in e.user_message


@pytest.mark.parametrize("status,code,kind", [(400, "batch_too_large", "bad_request"), (401, "unauthorized", "auth"), (403, "x", "forbidden"),
    (413, "payload_too_large", "too_large"), (429, "rate_limited", "rate_limit"), (500, "internal_error", "server"),
    (502, None, "server"), (503, "auth_not_configured", "config")])
@responses.activate
def test_status_mapping(c, status, code, kind):
    responses.post(f"{BASE}/v1/chat", status=status, headers={"Retry-After": "7"}, json={"error": {"code": code, "message": "m", "request_id": "r"}})
    assert err(c.chat, "x").kind == kind


@responses.activate
def test_429_mentions_retry_after(c):
    responses.post(f"{BASE}/v1/chat", status=429, headers={"Retry-After": "12"}, json={"error": {"code": "rate_limited", "message": "m"}})
    assert "12 seconds" in err(c.chat, "x").user_message


@responses.activate
def test_non_json_error_body_is_safe(c):
    responses.post(f"{BASE}/v1/chat", status=500, body="<html>Traceback secret</html>")
    e = err(c.chat, "x")
    assert e.kind == "server" and "Traceback" not in e.user_message


# ---- transport failures --------------------------------------------------------------------------------------------------------
@responses.activate
def test_connection_error_hides_host(c):
    responses.post(f"{BASE}/v1/chat", body=requests.ConnectionError("secret-host.internal"))
    e = err(c.chat, "x")
    assert e.kind == "connection" and "secret-host" not in e.user_message and "Unable to connect to CortexHR" in e.user_message


@responses.activate
def test_timeout(c):
    responses.post(f"{BASE}/v1/chat", body=requests.Timeout())
    assert err(c.chat, "x").kind == "timeout"


# ---- malformed / empty / unexpected --------------------------------------------------------------------------------------------
@responses.activate
def test_invalid_json_empty_body_and_non_object(c):
    responses.post(f"{BASE}/v1/chat", body="<html>", status=200)
    assert err(c.chat, "x").kind == "bad_response"
    responses.replace(responses.POST, f"{BASE}/v1/chat", body="", status=200)
    assert err(c.chat, "x").kind == "bad_response"
    responses.replace(responses.POST, f"{BASE}/v1/chat", json=[1, 2], status=200)
    assert err(c.chat, "x").kind == "bad_response"


@responses.activate
def test_missing_required_fields_rejected(c):
    responses.post(f"{BASE}/v1/attrition/predict", json={"model_version": "v", "n_records": 1, "results": [{"foo": 1}]})
    assert err(c.attrition_predict, [{}]).kind == "bad_response"
    responses.post(f"{BASE}/v1/forecast/headcount", json={"mode": "live", "trained_through": "x", "forecast": [{"month": "2026-10-01"}]})
    assert err(c.forecast).kind == "bad_response"


@responses.activate
def test_ready_returns_body_on_503(c):
    responses.get(f"{BASE}/v1/ready", status=503, json={"ready": False, "blocking": ["attrition"], "degraded": []})
    assert c.ready()["blocking"] == ["attrition"]


def test_oversized_request_refused_before_sending(c):
    e = err(c.resume_rank, {"job_title": "x"}, [{"resume_id": f"C-{i}", "resume_text": "x" * 20_000} for i in range(60)])
    assert e.kind == "too_large"


def test_nan_cannot_be_sent(c):
    with pytest.raises(ValueError):
        c.attrition_predict([{"tenure_months": float("nan")}])


@responses.activate
def test_logs_contain_no_keys_or_payload(c, caplog):
    caplog.set_level(logging.INFO, logger="cortexhr.console")
    responses.post(f"{BASE}/v1/chat", json=real("chat_faq"))
    c.chat("my confidential question about EMP-0188", "s")
    assert "KEY-SECRET" not in caplog.text and "EMP-0188" not in caplog.text and "confidential" not in caplog.text


# ---- monitoring (real captured bodies) -----------------------------------------------------------------------------------------------
import json
from pathlib import Path
MON = {k: v["body"] for k, v in json.loads((Path(__file__).parent / "fixtures" / "monitoring_samples.json").read_text()).items()}


@pytest.mark.parametrize("name", ["ok", "watch", "alert_drift", "alert_fairness", "alert_missing", "stale_ok"])
@responses.activate
def test_monitoring_real_bodies_accepted(c, name):
    responses.get(f"{BASE}/v1/monitoring", json=MON[name])
    r = c.monitoring()
    assert r["status"] in ("OK", "WATCH", "ALERT") and set(r["fairness"]) == {"twin_resume", "pay_gap"}
    assert "X-Monitoring-Publish-Key" not in responses.calls[0].request.headers and responses.calls[0].request.method == "GET"


@pytest.mark.parametrize("mutate", [lambda d: d.update(status="GREEN"), lambda d: d.pop("fairness"), lambda d: d["fairness"].pop("pay_gap"),
                                    lambda d: d["fairness"]["pay_gap"].update(status="FINE"), lambda d: d["drift"].update(watch="1"),
                                    lambda d: d.update(stale="no"), lambda d: d.update(reasons=["x"]), lambda d: d.pop("last_run")])
@responses.activate
def test_monitoring_malformed_bodies_rejected(c, mutate):
    body = json.loads(json.dumps(MON["ok"])); mutate(body)
    responses.get(f"{BASE}/v1/monitoring", json=body)
    assert err(c.monitoring).kind == "bad_response"


@responses.activate
def test_monitoring_404_and_unreadable(c):
    responses.get(f"{BASE}/v1/monitoring", status=404, json={"error": {"code": "monitoring_report_not_found", "message": "No monitoring report has been published to this server yet.", "request_id": "r"}})
    e = err(c.monitoring); assert e.kind == "not_found" and "published" in e.user_message
    responses.replace(responses.GET, f"{BASE}/v1/monitoring", status=503, json={"error": {"code": "monitoring_report_invalid", "message": "secret path", "request_id": "r"}})
    e = err(c.monitoring); assert e.kind == "unavailable" and "secret" not in e.user_message


def test_client_has_no_write_path_for_monitoring(c):
    assert not [m for m in dir(c) if "publish" in m.lower() or "put" in m.lower().split("_")]
