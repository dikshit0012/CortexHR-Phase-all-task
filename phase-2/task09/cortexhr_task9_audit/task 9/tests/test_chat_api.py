"""API tests for POST /v1/chat on the REAL Task 8 FastAPI app (app.main.create_app) plus regression checks that the Task 8 routes
are still registered and behave as before.

These tests need the Task 8 app to be importable, which needs the Task 5-7 source folders (see task 8/app/_paths.py:
set CORTEXHR_TASKS_ROOT if they are not next to `task 8`). If they cannot be found the whole module is SKIPPED with the reason.
The Task 8 anomaly SCORING itself is covered by Task 8's own test-suite, which needs the reference workbook; run it separately.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK8 = ROOT.parent / "task 8"
os.environ.setdefault("CORTEXHR_TASK9_ROOT", str(ROOT))
if not (TASK8 / "app").is_dir():
    pytest.skip(f"Task 8 folder not found next to task 9 ({TASK8.name})", allow_module_level=True)
sys.path.insert(0, str(TASK8))
try:
    from app.main import create_app
except RuntimeError as exc:                                         # _paths.py: Task 5-7 source folders not found
    pytest.skip(f"Task 8 app cannot be imported here: {exc}", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from cortexhr_task9.chat_service import GENERIC_FALLBACK, NO_ANSWER_FALLBACK, ChatService  # noqa: E402
from cortexhr_task9.faq import load_faq  # noqa: E402
from cortexhr_task9.paths import DEFAULT_FAQ, ENV_FAQ, ENV_MODEL  # noqa: E402
from conftest import FakeModel  # noqa: E402

FAQ = load_faq(DEFAULT_FAQ)
TASK8_ROUTES = {"/v1/health", "/v1/models", "/v1/anomaly/attendance", "/v1/anomaly/payroll",
                "/v1/anomaly/attendance/batch", "/v1/anomaly/payroll/batch"}


class FakeAnomalyService:
    """Test double for the Task 8 AnomalyService (the real one needs the reference workbook)."""
    max_batch_size = 100
    attendance = SimpleNamespace(version="attendance-anomaly-v1")
    payroll = SimpleNamespace(version="payroll-anomaly-v1")

    def info(self):
        return {"max_batch_size": 100, "detectors": {"attendance": {"artifact_version": "attendance-anomaly-v1"},
                                                     "payroll": {"artifact_version": "payroll-anomaly-v1"}}}


@pytest.fixture()
def client(service):
    with TestClient(create_app(service=FakeAnomalyService(), chat_service=service)) as c:
        yield c


def make_client(chat_service):
    return TestClient(create_app(service=FakeAnomalyService(), chat_service=chat_service))


# ---- /v1/chat ----------------------------------------------------------------------------------------------------------
def test_supported_question_returns_documented_schema_and_approved_answer(client):
    r = client.post("/v1/chat", json={"message": "How long is the notice period?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"intent", "confidence", "reply", "fallback"}
    assert body["intent"] == "notice_period_policy" and body["fallback"] is False
    assert body["reply"] == FAQ["FAQ-18"].answer
    assert isinstance(body["confidence"], float) and 0.0 <= body["confidence"] <= 1.0


def test_informal_question_with_typos(client):
    body = client.post("/v1/chat", json={"message": "plesae  whats the work from home policy??"}).json()
    assert body["intent"] == "wfh_policy" and body["reply"] == FAQ["FAQ-21"].answer and body["fallback"] is False


def test_uncertain_question_gets_the_polite_fallback(client):
    body = client.post("/v1/chat", json={"message": "can I bring my dog to the office"}).json()
    assert set(body) == {"intent", "confidence", "reply", "fallback"}
    assert body["intent"] is None and body["fallback"] is True and body["reply"] == GENERIC_FALLBACK
    assert 0.0 <= body["confidence"] < 0.64


def test_low_confidence_with_a_stub_model(responder):
    c = make_client(ChatService(FakeModel(confidence=0.2), responder))
    with c:
        body = c.post("/v1/chat", json={"message": "what is the work from home policy"}).json()
    assert body == {"intent": None, "confidence": 0.2, "reply": GENERIC_FALLBACK, "fallback": True}


@pytest.mark.parametrize("message", ["", " ", "     ", "\n\t "])
def test_empty_and_whitespace_messages_are_handled_safely(client, message):
    r = client.post("/v1/chat", json={"message": message})
    assert r.status_code == 200
    assert r.json() == {"intent": None, "confidence": None, "reply": GENERIC_FALLBACK, "fallback": True}


def test_recognised_intent_without_faq_returns_intent_and_fallback(client):
    body = client.post("/v1/chat", json={"message": "show my payslip for June"}).json()
    assert body["intent"] == "payslip_status" and body["fallback"] is True
    assert body["reply"].startswith(NO_ANSWER_FALLBACK) and body["reply"].endswith(FAQ["FAQ-28"].answer)


@pytest.mark.parametrize("payload", [{}, {"msg": "hi"}, {"message": None}, {"message": 123}, {"message": ["a"]},
                                     {"message": "hi", "extra": 1}, {"message": "x" * 1001}])
def test_invalid_requests_are_rejected_with_422_and_never_echoed(client, payload):
    r = client.post("/v1/chat", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail and all(set(e) == {"loc", "type", "msg"} for e in detail)
    assert "xxxxxxxx" not in r.text                                          # submitted values are not echoed back


def test_non_json_body_is_rejected(client):
    assert client.post("/v1/chat", content=b"not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/v1/chat").status_code == 422


def test_inference_error_returns_a_safe_fallback_without_details(responder):
    c = make_client(ChatService(FakeModel(error=RuntimeError("boom /secret/path password=hunter2")), responder))
    with c:
        r = c.post("/v1/chat", json={"message": "what is the notice period"})
    assert r.status_code == 200 and r.json()["fallback"] is True and r.json()["intent"] is None
    assert "hunter2" not in r.text and "secret" not in r.text and "Traceback" not in r.text


def test_chat_is_503_when_the_model_artifact_is_missing_but_other_endpoints_still_work(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_MODEL, str(tmp_path / "missing.joblib"))
    with TestClient(create_app(service=FakeAnomalyService())) as c:           # chat_service=None -> loaded from the (missing) artifact
        r = c.post("/v1/chat", json={"message": "hello"})
        assert r.status_code == 503 and "not loaded" in r.json()["detail"] and str(tmp_path) not in r.text
        assert c.get("/v1/health").status_code == 200 and c.get("/v1/models").status_code == 200


def test_model_is_loaded_once_at_startup_and_never_retrained_per_request(monkeypatch, artifact_path):
    import joblib
    from sklearn.pipeline import Pipeline
    monkeypatch.setenv(ENV_MODEL, str(artifact_path))
    monkeypatch.setenv(ENV_FAQ, str(DEFAULT_FAQ))
    real_load, loads = joblib.load, []
    monkeypatch.setattr(joblib, "load", lambda *a, **k: (loads.append(1), real_load(*a, **k))[1])
    def no_fit(*a, **k):
        raise AssertionError("fit must not be called by the API")
    monkeypatch.setattr(Pipeline, "fit", no_fit)
    with TestClient(create_app(service=FakeAnomalyService())) as c:           # real startup path: loads from the artifact
        assert len(loads) == 1
        for message in ["what is the work from home policy", "hello", "asdfgh", "how long is the probation period", ""]:
            assert c.post("/v1/chat", json={"message": message}).status_code == 200
        assert len(loads) == 1                                                  # still exactly one load after 5 requests


def test_chat_needs_no_authentication(client):
    assert client.post("/v1/chat", json={"message": "hello"}).status_code == 200


# ---- regression: Task 8 behaviour is preserved -----------------------------------------------------------------------------
def test_all_task8_routes_are_still_registered_and_only_chat_was_added(client):
    paths = {p for p in client.get("/openapi.json").json()["paths"] if p.startswith("/v1")}    # OpenAPI is stable across FastAPI versions
    assert TASK8_ROUTES <= paths
    assert paths - TASK8_ROUTES == {"/v1/chat"}


def test_task8_health_and_models_unchanged(client):
    h = client.get("/v1/health")
    assert h.status_code == 200 and h.json() == {"status": "ok", "service": "cortexhr-anomaly", "models_loaded": True}
    m = client.get("/v1/models")
    assert m.status_code == 200 and set(m.json()) == {"max_batch_size", "detectors"}


def test_task8_validation_behaviour_unchanged(client):
    r = client.post("/v1/anomaly/attendance", json={"employee_id": "E-SECRET-1"})
    assert r.status_code == 422 and "E-SECRET-1" not in r.text                # still no echo of submitted values
    assert all(set(e) == {"loc", "type", "msg"} for e in r.json()["detail"])
    assert client.post("/v1/anomaly/payroll/batch", json={"records": []}).status_code == 422


def test_task8_endpoints_still_503_when_their_service_is_not_loaded():
    app = create_app(service=FakeAnomalyService())
    c = TestClient(app)                                                       # no `with`: lifespan not run, nothing loaded
    assert c.get("/v1/health").status_code == 503
    assert c.post("/v1/chat", json={"message": "hi"}).status_code == 503
