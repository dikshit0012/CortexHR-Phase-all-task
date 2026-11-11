"""HTTP layer (#6, #7 at API level) and regression checks for Tasks 5-9 (#31-#33)."""
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cortexhr_task10.api import attach, router
from cortexhr_task10.config import TASK10_ROOT, Settings
from cortexhr_task10.service_client import AiServiceClient, HRServiceClient
from support import REF, ScriptedClassifier, make_env

REPO = TASK10_ROOT.parent


@pytest.fixture
def api(tmp_path):
    env = make_env(tmp_path, ScriptedClassifier({"show my payslip for march": ("payslip_status", 0.95), "hi": ("greet", 0.95),
                                                 "and last month?": ("out_of_scope", 0.1)}))
    app = FastAPI(); app.include_router(router)
    settings = Settings(reference_date=REF, demo_users_csv=tmp_path / "mock_data" / "demo_users.csv")
    attach(app, settings, classifier=env.manager.classifier, hr_client=HRServiceClient(None, client=env.hr), ai_client=AiServiceClient(None, client=env.ai))
    return TestClient(app), env


def post(c, text, user=None, sid=None, **extra):
    h = {"X-Demo-User": user} if user else {}
    return c.post("/v1/chat/converse", json={"message": text, **({"session_id": sid} if sid else {}), **extra}, headers=h)


def test_api_roundtrip_with_memory(api):
    c, env = api
    a = post(c, "show my payslip for March", "EMP-1001").json()
    b = post(c, "and last month?", "EMP-1001", a["session_id"]).json()
    assert a["action"] == "answered" and b["followup"] and b["slots"] == {"month": "2026-03"} and len(env.hr.log) == 2


def test_unknown_or_malformed_demo_user_is_401_and_makes_no_call(api):                 # #6
    c, env = api
    for bad in ("EMP-0000", "EMP-1001; DROP", "x" * 50):
        r = post(c, "show my payslip for March", bad)
        assert r.status_code == 401 and bad not in r.text
    assert env.hr.log == []


def test_no_header_is_anonymous_public_only(api):
    c, env = api
    assert post(c, "hi").json()["intent"] == "greet"
    assert post(c, "show my payslip for March").json()["action"] == "denied" and env.hr.log == []


def test_role_or_user_in_the_body_is_rejected_not_obeyed(api):                         # #4/#6
    c, env = api
    assert post(c, "hi", "EMP-1001", role="ADMIN").status_code == 422
    assert post(c, "hi", "EMP-1001", x_demo_user="EMP-2001").status_code == 422


@pytest.mark.parametrize("sid", ["EMP-1001", "short", "bad id with spaces 12345", "a@b.co" + "x" * 20])
def test_session_ids_that_look_like_employee_ids_or_emails_are_rejected(api, sid):
    c, _ = api
    assert post(c, "hi", "EMP-1001", sid).status_code == 422


def test_second_user_cannot_use_the_first_users_session_over_http(api):                # #7
    c, env = api
    a = post(c, "show my payslip for March", "EMP-1001").json()
    b = post(c, "and last month?", "EMP-1002", a["session_id"]).json()
    assert b["new_session"] and b["session_id"] != a["session_id"] and b["action"] == "fallback" and len(env.hr.log) == 1


def test_503_when_task10_chat_is_not_loaded():
    app = FastAPI(); app.include_router(router)
    attach(app, Settings(responses_csv=Path("/nonexistent.csv")))
    assert post(TestClient(app), "hi").status_code == 503


# ---------------- regression (#31-#33) ----------------
def test_31_task9_chat_service_answers_exactly_as_before(real_classifier):
    sys.path.insert(0, str(REPO / "task09" / "cortexhr_task9" / "task 9" / "src"))
    from cortexhr_task9.chat_service import load_chat_service
    r = load_chat_service().reply("How long is the notice period?")
    assert r.intent == "notice_period_policy" and abs(r.confidence - 0.9458) < 0.005 and not r.fallback       # value published in Task 9's own evidence


def test_31b_task10_changes_exactly_one_file_inside_task9():
    pristine = Path("/home/claude/work/t9/task 9")
    if not pristine.exists():
        pytest.skip("pristine Task 9 copy not available")
    cur = REPO / "task09" / "cortexhr_task9" / "task 9"
    def digest(root):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
    a, b = digest(cur), digest(pristine)
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert changed == {"tests/test_chat_api.py"}          # the ONE deliberate change: the route-set assertion now allows /v1/chat/converse


@pytest.mark.skipif(not __import__("os").environ.get("RUN_PRIOR_TASK_SUITES"), reason="slow (~2 min); set RUN_PRIOR_TASK_SUITES=1, or run each prior task's `pytest tests` folder directly (see README)")
@pytest.mark.parametrize("task", [5, 6, 7])
def test_33_tasks_5_to_7_suites_are_untouched_and_pass(task):
    root = Path("/home/claude/up/CortexHR--master__1_/CortexHR--master") / f"task {task}"
    if not root.exists():
        pytest.skip("repository copy with Tasks 5-7 not available")
    p = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-x", "-p", "no:cacheprovider", "-W", "ignore"], cwd=root, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-400:]
