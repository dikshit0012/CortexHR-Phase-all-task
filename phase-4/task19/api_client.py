"""The ONLY module that talks to the CortexHR API. Routes, request bodies and response fields are exactly those of the
Task 17 service (cortexhr_api). Failures raise ApiError with a message that is safe for an HR manager; a failed call never
returns made-up data."""
from __future__ import annotations

import json as _json
import logging
import time

import requests

from config import Config

log = logging.getLogger("cortexhr.console")
MON_OVERALL = ("OK", "WATCH", "ALERT")
MON_CHECK = ("OK", "WATCH", "ALERT", "NOT_EVALUATED", "ERROR")
MON_DRIFT_KEYS = ("features_checked", "ok", "watch", "alert", "not_evaluated", "error")
MAX_BODY_BYTES = 950_000          # the API rejects bodies over 1,000,000 bytes (413)


class ApiError(Exception):
    def __init__(self, kind: str, user_message: str, status: int | None = None, code: str | None = None,
                 request_id: str | None = None):
        super().__init__(user_message)
        self.kind, self.user_message, self.status, self.code, self.request_id = kind, user_message, status, code, request_id


def _error_body(resp) -> dict:
    try:
        e = resp.json().get("error") or {}
        return e if isinstance(e, dict) else {}
    except Exception:
        return {}


def _clean(msg: str) -> str:
    """The API's validation/400 messages name fields and rules but never echo values."""
    return (msg or "").replace("Invalid request: ", "").strip()[:300]


class ApiClient:
    def __init__(self, config: Config, session: requests.Session | None = None):
        self.config, self._s = config, session or requests.Session()

    # ------------------------------------------------------------------ core
    def _call(self, method: str, path: str, *, label: str, body=None, allow_503_body: bool = False):
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        data = None
        if body is not None:
            data = _json.dumps(body, allow_nan=False).encode()
            if len(data) > MAX_BODY_BYTES:
                raise ApiError("too_large", "That is too much data for one request. Please send fewer rows or shorter texts.")
            headers["Content-Type"] = "application/json"
        t0 = time.monotonic()
        try:
            resp = self._s.request(method, self.config.api_url + path, data=data, headers=headers, timeout=self.config.timeout)
        except requests.Timeout:
            log.warning("api %s timeout", label)
            raise ApiError("timeout", "CortexHR is taking too long to respond. If it was idle it may be waking up: please try again in a minute.")
        except requests.RequestException:
            log.warning("api %s connection_error", label)
            raise ApiError("connection", "Unable to connect to CortexHR. Please try again in a moment.")
        log.info("api %s status=%s ms=%d", label, resp.status_code, (time.monotonic() - t0) * 1000)   # never bodies, ids or keys
        return self._handle(resp, allow_503_body)

    @staticmethod
    def _handle(resp, allow_503_body=False):
        s, e = resp.status_code, _error_body(resp)
        rid, code = e.get("request_id"), e.get("code")
        if 200 <= s < 300 or (allow_503_body and s == 503 and not code):
            if not resp.content:
                raise ApiError("bad_response", "CortexHR returned an empty response.", s, request_id=rid)
            try:
                data = resp.json()
            except ValueError:
                raise ApiError("bad_response", "CortexHR returned a response the console could not read.", s)
            if not isinstance(data, dict) or not data:
                raise ApiError("bad_response", "CortexHR returned an unexpected response.", s)
            return data
        if s == 400:
            raise ApiError("bad_request", _clean(e.get("message")) or "Your request could not be processed. Please check the input.", s, code, rid)
        if s == 401:
            raise ApiError("auth", "The console's access key is missing or was not accepted. Please contact the administrator.", s, code, rid)
        if s == 403:
            raise ApiError("forbidden", "You do not have permission to view this information.", s, code, rid)
        if s == 404:
            raise ApiError("not_found", _clean(e.get("message")) or "No matching resource was found.", s, code, rid)
        if s == 413:
            raise ApiError("too_large", "That is too much data for one request. Please send fewer rows or shorter texts.", s, code, rid)
        if s == 422:
            detail = _clean(e.get("message"))
            raise ApiError("validation", "Your request could not be processed. Please check the input." + (f" Details: {detail}" if detail else ""), s, code, rid)
        if s == 429:
            wait = resp.headers.get("Retry-After")
            raise ApiError("rate_limit", "Too many requests right now." + (f" Please wait about {wait} seconds and try again." if wait and wait.isdigit() else " Please wait a moment and try again."), s, code, rid)
        if s == 503 and code == "model_unavailable":
            raise ApiError("unavailable", "This capability is not available on the server right now. Please contact the administrator.", s, code, rid)
        if s == 503 and code == "monitoring_report_invalid":
            raise ApiError("unavailable", "The latest monitoring report is unreadable. The data science owner needs to re-run the monitor.", s, code, rid)
        if s == 503 and code == "auth_not_configured":
            raise ApiError("config", "The service is not fully configured yet. Please contact the administrator.", s, code, rid)
        if s >= 500:
            raise ApiError("server", "The service is temporarily unavailable.", s, code, rid)
        raise ApiError("http", "Your request could not be processed. Please check the input.", s, code, rid)

    @staticmethod
    def _need(data: dict, keys, what: str, list_key: str | None = None, item_keys=()):
        ok = all(k in data for k in keys)
        if ok and list_key:
            items = data[list_key]
            ok = isinstance(items, list) and all(isinstance(i, dict) and all(k in i for k in item_keys) for i in items)
        if not ok:
            raise ApiError("bad_response", f"CortexHR returned an unexpected {what} response.")
        return data

    # ------------------------------------------------------------------ routes (Task 17)
    def health(self) -> dict:
        return self._need(self._call("GET", "/v1/health", label="health"), ["application", "models", "models_available", "models_unavailable"], "health")

    def ready(self) -> dict:
        return self._need(self._call("GET", "/v1/ready", label="ready", allow_503_body=True), ["ready"], "readiness")

    def attrition_predict(self, records: list[dict]) -> dict:
        d = self._call("POST", "/v1/attrition/predict", label="attrition", body={"records": records})
        return self._need(d, ["model_version", "n_records", "results"], "attrition", "results", ["attrition_probability_12m"])

    def anomaly(self, kind: str, records: list[dict]) -> dict:
        if kind not in ("attendance", "payroll"):
            raise ValueError(kind)
        d = self._call("POST", f"/v1/anomaly/{kind}", label=f"anomaly_{kind}", body={"records": records})
        return self._need(d, ["model_version", "n_records", "n_flagged", "results"], "anomaly", "results",
                          ["result_id", "employee_id", "period", "anomaly_score", "flagged"])

    def chat(self, message: str, session_id: str | None = None) -> dict:
        body = {"message": message}
        if session_id:
            body["session_id"] = session_id
        return self._need(self._call("POST", "/v1/chat", label="chat", body=body), ["reply", "fallback"], "chat")

    def resume_rank(self, job: dict, resumes: list[dict]) -> dict:
        d = self._call("POST", "/v1/resume/rank", label="resume", body={"job": job, "resumes": resumes})
        return self._need(d, ["ranking", "n_resumes"], "resume ranking", "ranking", ["rank", "resume_id", "final_score"])

    def payband(self, profile: dict) -> dict:
        return self._need(self._call("POST", "/v1/payband/analyze", label="payband", body=profile),
                          ["predicted_salary_mid", "band_status", "comparison"], "pay-band")

    def forecast(self, department_id: str = "ALL", horizon_months: int = 6) -> dict:
        d = self._call("POST", "/v1/forecast/headcount", label="forecast",
                       body={"department_id": department_id, "horizon_months": horizon_months})
        return self._need(d, ["mode", "forecast", "trained_through"], "forecast", "forecast",
                          ["month", "expected_headcount", "lower_80", "upper_80"])

    def monitoring(self) -> dict:
        """GET /v1/monitoring (Task 18): the latest COMPLETED report, reduced by the API to whitelisted aggregates. Read-only: the console
        never runs monitoring and never holds the publish key. 404 = nothing published yet; 503 = the stored report is unreadable."""
        d = self._call("GET", "/v1/monitoring", label="monitoring")
        self._need(d, ["status", "timestamp", "execution_status", "execution_ok", "evidence_complete", "drift", "fairness", "reasons",
                       "last_run", "report_age_hours", "stale"], "monitoring")
        ok = (d["status"] in MON_OVERALL and isinstance(d["drift"], dict) and set(MON_DRIFT_KEYS) <= set(d["drift"])
              and all(isinstance(d["drift"][k], int) for k in MON_DRIFT_KEYS) and isinstance(d["fairness"], dict)
              and all(isinstance(d["fairness"].get(k), dict) and d["fairness"][k].get("status") in MON_CHECK for k in ("twin_resume", "pay_gap"))
              and isinstance(d["reasons"], list) and all(isinstance(r, dict) and {"category", "check", "item", "status"} <= set(r) for r in d["reasons"])
              and isinstance(d["stale"], bool) and isinstance(d["execution_ok"], bool) and isinstance(d["evidence_complete"], bool))
        if not ok:
            raise ApiError("bad_response", "CortexHR returned an unexpected monitoring response.")
        return d
