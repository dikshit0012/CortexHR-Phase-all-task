"""CortexHR production API: one versioned FastAPI application for all models.

Run:  uvicorn cortexhr_api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api.v1.router import api_v1
from .config import API_VERSION, Settings
from .registry import ModelRegistry
from .security import ApiError

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
log = logging.getLogger("cortexhr")

TAGS = [
    {"name": "Health", "description": "Liveness, model availability and readiness (public)."},
    {"name": "Attrition", "description": "Probability that an employee leaves within 12 months (attrition-v2)."},
    {"name": "Anomaly Detection", "description": "Attendance and payroll anomalies with reason codes and money at risk."},
    {"name": "HR Chatbot", "description": "FAQ chatbot with confidence-based fallback. No access to employee records."},
    {"name": "Resume Ranking", "description": "Explainable ranking of resumes against a job description."},
    {"name": "Pay-Band Analysis", "description": "Recommended pay for a role profile (decision support only)."},
    {"name": "Headcount Forecast", "description": "Six-month headcount forecast with 80% prediction interval."},
]


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Only whitelisted fields are emitted, so request bodies can never leak into logs."""
    FIELDS = ("request_id", "method", "path", "status", "duration_ms", "client_error", "model", "event")

    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"), "level": record.levelname, "logger": record.name,
               "msg": record.getMessage()}
        out.update({k: getattr(record, k) for k in self.FIELDS if hasattr(record, k)})
        return json.dumps(out, ensure_ascii=False)


def configure_logging() -> None:
    root = logging.getLogger()
    if any(getattr(h, "_cortexhr", False) for h in root.handlers):
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    h._cortexhr = True  # type: ignore[attr-defined]
    root.addHandler(h)
    root.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").disabled = True          # we log requests ourselves (no query strings, no bodies)


class _TooLarge(Exception):
    pass


class BodySizeLimit:
    """Pure-ASGI limit on the request body (413), enforced on Content-Length and on the bytes actually received."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            return await self._reject(send)
        received, started = 0, False

        async def limited_receive():
            nonlocal received
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLarge()
            return msg

        async def tracking_send(message):
            nonlocal started
            started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _TooLarge:
            if not started:
                await self._reject(send)

    async def _reject(self, send):
        body = json.dumps({"error": {"code": "payload_too_large", "message": f"Request body exceeds {self.max_bytes} bytes."}}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class RateLimiter:
    """In-memory sliding window per client. Adequate for one instance; use a shared store if the service is scaled out."""

    def __init__(self, per_minute: int):
        self.per_minute, self.hits = per_minute, defaultdict(deque)

    def check(self, client: str) -> Optional[int]:
        now, q = time.monotonic(), self.hits[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return max(1, int(60 - (now - q[0])))
        q.append(now)
        return None


def _error(status: int, code: str, message: str, request: Optional[Request] = None, headers: Optional[dict] = None) -> JSONResponse:
    rid = getattr(request.state, "request_id", None) if request is not None else None
    return JSONResponse({"error": {"code": code, "message": message, "request_id": rid}}, status_code=status, headers=headers)


def create_app(settings: Optional[Settings] = None, registry: Optional[ModelRegistry] = None) -> FastAPI:
    configure_logging()
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started_at = time.time()
        log.info("startup: loading models", extra={"event": "startup"})
        app.state.registry = registry if registry is not None else ModelRegistry().load_all(settings)
        missing = app.state.registry.missing()
        log.info("startup complete; unavailable models: %s", missing or "none", extra={"event": "startup_complete"})
        if settings.require_auth and not settings.api_keys:
            log.warning("CORTEXHR_API_KEYS is not set: protected endpoints will answer 503 until a key is configured")
        yield
        log.info("shutdown", extra={"event": "shutdown"})

    app = FastAPI(
        title="CortexHR API", version=API_VERSION, lifespan=lifespan, openapi_tags=TAGS,
        docs_url="/docs" if settings.docs_enabled else None, redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        description=("One versioned API for the CortexHR models: attrition, attendance and payroll anomalies, HR chatbot, resume ranking, "
                     "pay-band analysis and headcount forecasting. Prediction endpoints require an `X-API-Key` header. "
                     "Outputs are decision support for HR professionals, not automatic decisions."))
    app.state.settings = settings
    app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
    app.state.started_at = time.time()
    app.state.registry = registry if registry is not None else ModelRegistry()

    app.add_middleware(BodySizeLimit, max_bytes=settings.max_body_bytes)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "X-API-Key"], max_age=600)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex[:16]
        if not all(c.isalnum() or c in "-_" for c in rid):
            rid = uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.perf_counter()
        path = request.url.path
        limited = path.startswith("/v1/") and path not in ("/v1/health", "/v1/ready")
        if limited:
            fwd = request.headers.get("x-forwarded-for", "")
            client = fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")
            wait = request.app.state.limiter.check(client)
            if wait is not None:
                resp = _error(429, "rate_limited", "Too many requests. Please retry later.", request, {"Retry-After": str(wait)})
                _finish(request, resp, t0)
                return resp
        try:
            resp = await call_next(request)
        except Exception:                                                     # last-resort guard; details go to logs only
            log.exception("unhandled error", extra={"request_id": rid})
            resp = _error(500, "internal_error", "An unexpected error occurred.", request)
        _finish(request, resp, t0)
        return resp

    def _finish(request: Request, resp, t0: float) -> None:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        resp.headers["X-Request-ID"] = request.state.request_id
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Cache-Control", "no-store")
        if request.url.path.startswith("/app"):
            resp.headers["Cache-Control"] = "no-cache"
        log.info("request", extra={"request_id": request.state.request_id, "method": request.method, "path": request.url.path,
                                   "status": resp.status_code, "duration_ms": ms})

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return _error(exc.status, exc.code, exc.message, request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Report WHERE and WHY, never the submitted value (it may be sensitive).
        issues = [f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()][:10]
        return _error(422, "validation_error", "Invalid request: " + "; ".join(issues), request)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, "Resource not found." if exc.status_code == 404 else str(exc.detail), request)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.error("unhandled %s", type(exc).__name__, extra={"request_id": getattr(request.state, "request_id", None)})
        return _error(500, "internal_error", "An unexpected error occurred.", request)

    app.include_router(api_v1)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/", include_in_schema=False)
    def root():
        if FRONTEND_DIR.is_dir():
            return RedirectResponse(url="/app/", status_code=307)
        return {"name": "CortexHR API", "version": API_VERSION,
                "description": "AI-powered HR analytics: attrition, anomalies, chatbot, resume ranking, pay bands and headcount forecast.",
                "links": {"dashboard": "/app/", "documentation": "/docs", "openapi": "/openapi.json", "health": "/v1/health", "ready": "/v1/ready"},
                "authentication": "Send your key in the X-API-Key header for prediction endpoints."}

    if FRONTEND_DIR.is_dir():
        app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
