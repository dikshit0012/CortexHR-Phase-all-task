"""Test environment: the REAL mock_server.py (unmodified copy) running on SYNTHETIC mock_data, a recording HTTP layer so tests can
assert which calls did or did not happen, and a scripted classifier for forcing misclassifications.

mock_data/*.csv were not supplied with the starter files. The tables below are SYNTHETIC; only the columns that mock_server.py itself
reads (employee_id, manager_id, run_id, period_end_date, applied_on, role) are certain. Pay values are sentinel strings so tests can
prove they never appear in a reply."""
from __future__ import annotations

import importlib.util
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from fastapi.testclient import TestClient

from cortexhr_task10.classifier import Prediction
from cortexhr_task10.config import TASK10_ROOT
from cortexhr_task10.dialogue import DialogueManager
from cortexhr_task10.identity import DemoUserDirectory, Identity
from cortexhr_task10.people import EmployeeDirectory, EmployeeRecord
from cortexhr_task10.policy import load_policies
from cortexhr_task10.service_client import AiServiceClient, HRServiceClient
from cortexhr_task10.sessions import SessionStore

REF = date(2026, 4, 15)
MOCK_SRC = TASK10_ROOT / "starter" / "mock_services" / "mock_server.py"
RESPONSES = TASK10_ROOT / "data" / "chatbot_responses.csv"
PAY_SENTINELS = {"EMP-1001": "NETPAY-SENTINEL-1001", "EMP-1002": "NETPAY-SENTINEL-1002", "EMP-2001": "NETPAY-SENTINEL-2001"}
USERS = {"EMP-1001": "EMPLOYEE", "EMP-1002": "EMPLOYEE", "EMP-2001": "HR", "EMP-3001": "ADMIN", "EMP-9001": "INTERN"}   # INTERN = unknown role


def build_mock_data(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    ids = list(USERS)
    pd.DataFrame({"employee_id": ids, "role": list(USERS.values())}).to_csv(d / "demo_users.csv", index=False)
    pd.DataFrame({"employee_id": ids, "full_name": ["Asha Test", "Priya Nair", "Hema Lead", "Adam Admin", "Ian Intern"],
                  "role_id": ["R1"] * 5, "manager_id": ["EMP-2001", "EMP-2001", "", "", "EMP-2001"], "manager_name": ["Hema Lead"] * 2 + ["", "", "Hema Lead"],
                  "department_name": ["Engineering", "Sales", "HR", "Admin", "Engineering"]}).to_csv(d / "employees.csv", index=False)
    pd.DataFrame({"employee_id": ids, "cl": [7, 3, 9, 9, 1], "sl": [5, 4, 9, 9, 1], "el": [12, 6, 9, 9, 1]}).to_csv(d / "leave_balances.csv", index=False)
    pd.DataFrame({"employee_id": ["EMP-1001", "EMP-1002"], "applied_on": ["2026-03-01", "2026-03-02"], "dates": ["2026-03-10 to 2026-03-11"] * 2,
                  "status": ["approved", "pending"]}).to_csv(d / "leave_requests.csv", index=False)
    rows = [{"employee_id": e, "run_id": f"PR-2026-{m:02d}", "status": "ready", "net_pay": PAY_SENTINELS[e]} for e in PAY_SENTINELS for m in (1, 2, 3, 4)]
    pd.DataFrame(rows).to_csv(d / "payslips.csv", index=False)
    pd.DataFrame({"employee_id": ["EMP-1001", "EMP-1001", "EMP-1002"], "period_end_date": ["2026-03-31", "2026-02-28", "2026-03-31"],
                  "present": [20, 19, 18], "scheduled": [22, 20, 22], "late": [2, 1, 4]}).to_csv(d / "attendance_monthly.csv", index=False)
    pd.DataFrame({"date": ["2026-01-26", "2026-05-01", "2026-08-15"], "name": ["Republic Day", "Labour Day", "Independence Day"]}).to_csv(d / "holidays.csv", index=False)


def load_mock_app(tmp: Path):
    """Copy the UNMODIFIED mock_server.py next to the synthetic mock_data and import it (the server reads mock_data at import)."""
    shutil.copy(MOCK_SRC, tmp / "mock_server.py")
    build_mock_data(tmp / "mock_data")
    name = f"mock_server_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, tmp / "mock_server.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod.app


class RecordingClient(TestClient):
    """Real HTTP semantics against the real mock app; every request is logged as (method, path-with-query, X-Demo-User)."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.log: List[Tuple[str, str, Optional[str]]] = []

    def request(self, method, url, **kw):
        hdr = (kw.get("headers") or {}).get("X-Demo-User")
        q = kw.get("params") or {}
        self.log.append((method, str(url) + ("?" + "&".join(f"{k}={v}" for k, v in q.items()) if q else ""), hdr))
        return super().request(method, url, **kw)


class RecordingAI(httpx.Client):
    """Stands in for the /ai/v1 service (mock_server.py does not provide it). Canned, UNVERIFIED-shape JSON."""
    def __init__(self, replies: Optional[Dict[str, object]] = None, fail: Optional[Exception] = None):
        self.replies = replies or {}; self.fail = fail; self.log: List[Tuple[str, str, Optional[str]]] = []
        super().__init__(transport=httpx.MockTransport(self._h), base_url="http://ai.test")

    def _h(self, req: httpx.Request) -> httpx.Response:
        if self.fail:
            raise self.fail
        body = self.replies.get(req.url.path, {"count": 2, "days": ["2026-04-02", "2026-04-09"], "n": 3, "drivers": ["overtime", "tenure"], "reason": "duplicate",
                                                "low": 600, "high": 900, "median": 750, "value": 120, "month": "2026-Q3", "score": 80, "matched": ["sql"], "missing": ["spark"]})
        return httpx.Response(200, json=body)

    def request(self, method, url, **kw):
        self.log.append((method, str(url), (kw.get("headers") or {}).get("X-Demo-User")))
        return super().request(method, url, **kw)


class ScriptedClassifier:
    """Returns a chosen prediction per message (default: low-confidence out_of_scope). threshold matches Task 9."""
    threshold = 0.64

    def __init__(self, script: Optional[Dict[str, Tuple[str, float]]] = None):
        self.script = {k.lower(): v for k, v in (script or {}).items()}

    def classify(self, text: str) -> Prediction:
        i, c = self.script.get(text.lower(), ("out_of_scope", 0.10)); return Prediction(i, c)


DIRECTORY = EmployeeDirectory([EmployeeRecord("EMP-1002", "Priya Nair", "Sales"), EmployeeRecord("EMP-5001", "Rahul Sharma", "Engineering"),
                               EmployeeRecord("EMP-5002", "Rahul Sharma", "Finance"), EmployeeRecord("EMP-5003", "Karthik Iyer", "Operations")])


@dataclass
class Env:
    manager: DialogueManager
    hr: RecordingClient
    ai: RecordingAI
    sessions: SessionStore
    users: DemoUserDirectory

    def who(self, employee_id: str) -> Identity:
        return self.users.lookup(employee_id)

    def say(self, employee_id: str, text: str, sid: Optional[str] = None):
        return self.manager.handle(self.who(employee_id), sid, text)

    @property
    def hr_paths(self) -> List[str]:
        return [p for _, p, _ in self.hr.log]

    @property
    def protected_calls(self) -> int:
        return len(self.hr.log) + len(self.ai.log)


def make_env(tmp: Path, classifier, *, ref: date = REF, ai: Optional[RecordingAI] = None, hr_client: Optional[HRServiceClient] = None,
             directory: Optional[EmployeeDirectory] = DIRECTORY, ttl: float = 1800.0, clock: Callable[[], float] = None) -> Env:
    app = load_mock_app(tmp)
    rc = RecordingClient(app, base_url="http://hr.test")
    ai_c = ai or RecordingAI()
    store = SessionStore(ttl_seconds=ttl, **({"clock": clock} if clock else {}))
    mgr = DialogueManager(classifier=classifier, policies=load_policies(RESPONSES), hr=hr_client or HRServiceClient(None, client=rc),
                          ai=AiServiceClient(None, client=ai_c), sessions=store, today=lambda: ref, directory=directory)
    return Env(mgr, rc, ai_c, store, DemoUserDirectory(USERS))
