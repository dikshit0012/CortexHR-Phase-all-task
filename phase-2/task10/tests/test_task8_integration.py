"""The REAL Task 8 app (`task 8/app`, as supplied + the Task 9 patch + the Task 10 patch) with a stub scoring service.
NOT covered: loading the real model artifacts / reference workbook (the workbook 02_CortexHR_AI_ML_Dataset.xlsx and Python 3.14.6 that the
artifacts were built with are not available here) and the Task 8 test-suite itself (not supplied)."""
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cortexhr_task10.config import TASK10_ROOT

T8 = TASK10_ROOT.parent / "task08" / "task 8"
T9 = TASK10_ROOT.parent / "task09" / "cortexhr_task9" / "task 9"
pytestmark = pytest.mark.skipif(not (T8 / "app" / "main.py").exists(), reason="task 8/app not present")

RESULT = {"result_id": "AN-0001", "source_dataset": "attendance_monthly_features", "record_id": "ATT-1", "employee_id": "EMP-1", "period": "2026-03-01",
          "anomaly_score": 0.5, "flagged": "N", "reason_code": None, "reason_text": None, "money_at_risk": None, "model_version": "x", "scored_at": "2026-03-01 00:00:00"}
ATT = {"attendance_feature_id": "ATT-1", "employee_id": "EMP-1", "month": "2026-03-01", "scheduled_days": 22, "present_days": 20, "half_days_unpaid": 0,
       "paid_leave_days": 2, "unpaid_absence_days": 0, "longest_unpaid_streak_days": 0, "late_arrivals": 1, "avg_hours_worked": 8.0, "overtime_hours": 0.0, "attendance_rate": 0.9}


@pytest.fixture(scope="module")
def make_app():
    sys.path.insert(0, str(T8)); sys.path.insert(0, str(T9 / "src"))
    from app.main import create_app
    from cortexhr_task9.chat_service import load_chat_service
    chat = load_chat_service()
    stub = SimpleNamespace(attendance=SimpleNamespace(version="att"), payroll=SimpleNamespace(version="pay"), max_batch_size=100,
                           score_attendance=lambda recs: [dict(RESULT) for _ in recs], score_payroll=lambda recs: [dict(RESULT) for _ in recs],
                           info=lambda: {"max_batch_size": 100, "detectors": {}})
    return lambda: TestClient(create_app(service=stub, chat_service=chat))


def test_32_task8_endpoints_health_models_anomaly_still_work(make_app):
    with make_app() as c:
        assert c.get("/v1/health").json() == {"status": "ok", "service": "cortexhr-anomaly", "models_loaded": True}
        assert c.get("/v1/models").status_code == 200
        ok = c.post("/v1/anomaly/attendance", json=ATT)
        assert ok.status_code == 200 and ok.json()["result_id"] == "AN-0001"
        assert c.post("/v1/anomaly/attendance/batch", json={"records": [ATT]}).json()["count"] == 1
        bad = c.post("/v1/anomaly/attendance", json={**ATT, "present_days": -1, "employee_id": "EMP-SECRET-77", "extra_field": 1})
        assert bad.status_code == 422 and "EMP-SECRET-77" not in bad.text and '"input"' not in bad.text      # Task 8: invalid input is rejected without echoing values


def test_task9_chat_still_answers_beside_task10(make_app):
    with make_app() as c:
        r = c.post("/v1/chat", json={"message": "How long is the notice period?"}).json()
        assert r["intent"] == "notice_period_policy" and not r["fallback"]


def test_converse_is_registered_and_reuses_the_task9_model(make_app):
    with make_app() as c:
        assert "/v1/chat/converse" in c.get("/openapi.json").json()["paths"]
        r = c.post("/v1/chat/converse", json={"message": "hi"})
        assert r.status_code == 200 and r.json()["intent"] == "greet" and r.json()["action"] == "static"
        assert c.app.state.converse.classifier._model is c.app.state.chat.model          # same object: loaded once


def test_converse_fails_closed_without_a_demo_user_table(make_app):
    with make_app() as c:                                   # CORTEXHR_DEMO_USERS_CSV unset -> nobody can log in
        assert c.post("/v1/chat/converse", json={"message": "show my payslip for March"}, headers={"X-Demo-User": "EMP-1001"}).status_code == 401
        assert c.post("/v1/chat/converse", json={"message": "show my payslip for March"}).json()["action"] == "denied"


def test_anomaly_endpoints_survive_a_broken_converse_setup(monkeypatch, make_app):
    monkeypatch.setenv("CORTEXHR_DEMO_USERS_CSV", "/nonexistent/demo_users.csv")
    with make_app() as c:
        assert c.post("/v1/chat/converse", json={"message": "hi"}).status_code == 503
        assert c.get("/v1/health").status_code == 200 and c.post("/v1/anomaly/attendance", json=ATT).status_code == 200
        assert c.post("/v1/chat", json={"message": "hello"}).status_code == 200
