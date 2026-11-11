"""HTTP clients. Service-call logic only: no intent classification, no authorization decisions, no dialogue state.

HRServiceClient talks to the mock HR service (mock_server.py). Every method takes the AUTHENTICATED employee id and sends it as
`X-Demo-User`; there is deliberately no parameter through which a message, slot or session could name a different user, and the
`employee` query parameter is always the literal 'me' (exactly as chatbot_responses specifies).
AiServiceClient is the same idea for the `/ai/v1/...` calls listed in chatbot_responses. NOTE: mock_server.py does NOT serve those
routes and no contract for them was provided, so their request bodies and response keys are UNVERIFIED assumptions (see README).
Errors become ServiceError(kind) - the kind is a short code, never a response body, URL query, or payroll value."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

import httpx

from .identity import EMPLOYEE_ID_RE

log = logging.getLogger("cortexhr.task10.service")
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class ServiceError(Exception):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind                    # 'timeout' | 'unavailable' | 'unauthorized' | 'not_found' | 'http' | 'invalid'


class _BaseClient:
    def __init__(self, base_url: Optional[str], *, client: Optional[httpx.Client] = None, connect_timeout: float = 2.0, read_timeout: float = 5.0):
        if client is None and base_url:
            client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(read_timeout, connect=connect_timeout), follow_redirects=False)
        self._client = client

    def _request(self, method: str, path: str, employee_id: Optional[str], *, params: Optional[Dict[str, Any]] = None, json: Any = None) -> Any:
        if self._client is None:
            raise ServiceError("unavailable")
        headers = {}
        if employee_id is not None:                      # None is only used for the public holiday list (no identity needed)
            if not isinstance(employee_id, str) or not EMPLOYEE_ID_RE.match(employee_id):
                raise ServiceError("unauthorized")
            headers["X-Demo-User"] = employee_id
        try:
            resp = self._client.request(method, path, headers=headers, params=params, json=json)
        except httpx.TimeoutException:
            log.warning("service timeout: %s %s", method, path.split("?")[0]); raise ServiceError("timeout")
        except httpx.HTTPError:
            log.warning("service unavailable: %s", method); raise ServiceError("unavailable")
        if resp.status_code in (401, 403):
            raise ServiceError("unauthorized")
        if resp.status_code == 404:
            raise ServiceError("not_found")
        if resp.status_code >= 400:
            log.warning("service http %s", resp.status_code); raise ServiceError("http")
        try:
            return resp.json()
        except ValueError:
            raise ServiceError("invalid")


class HRServiceClient(_BaseClient):
    @staticmethod
    def _dict(data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ServiceError("invalid")
        return data

    @staticmethod
    def _need(employee_id: Optional[str]) -> None:
        if employee_id is None:
            raise ServiceError("unauthorized")           # a protected endpoint is never called without an authenticated id

    @staticmethod
    def _list(data: Any) -> list:
        if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
            raise ServiceError("invalid")
        return data

    def holidays(self, employee_id: Optional[str] = None) -> list:
        return self._list(self._request("GET", "/api/v1/holidays", employee_id))

    def employee_me(self, employee_id: str) -> dict:
        self._need(employee_id)
        return self._dict(self._request("GET", "/api/v1/employees/me", employee_id))

    def team(self, employee_id: str) -> list:
        self._need(employee_id)
        return self._list(self._request("GET", "/api/v1/employees/me/team", employee_id))

    def leave_balance(self, employee_id: str) -> dict:
        self._need(employee_id)
        return self._dict(self._request("GET", "/api/v1/leave/balance", employee_id, params={"employee": "me"}))

    def leave_requests(self, employee_id: str, limit: int = 1) -> list:
        self._need(employee_id)
        return self._list(self._request("GET", "/api/v1/leave/requests", employee_id, params={"employee": "me", "limit": int(limit)}))

    def payslip(self, employee_id: str, month: str) -> dict:
        self._need(employee_id)
        if not MONTH_RE.match(month or ""):
            raise ServiceError("invalid")                    # also keeps path injection out of /payslips/me/{month}
        return self._dict(self._request("GET", f"/api/v1/payslips/me/{month}", employee_id))

    def attendance(self, employee_id: str, month: str) -> dict:
        self._need(employee_id)
        if not MONTH_RE.match(month or ""):
            raise ServiceError("invalid")
        return self._dict(self._request("GET", "/api/v1/attendance/summary", employee_id, params={"employee": "me", "month": month}))


class AiServiceClient(_BaseClient):
    """`/ai/v1/...` routes named in chatbot_responses. Unverified request/response shapes - see module docstring."""

    def call(self, method: str, path: str, employee_id: str, *, params: Optional[Dict[str, Any]] = None, json: Any = None) -> dict:
        data = self._request(method, path, employee_id, params=params, json=json)
        if not isinstance(data, dict):
            raise ServiceError("invalid")
        return data
