"""API-key protection (fails closed) and helpers that keep messages free of filesystem paths."""
from __future__ import annotations

import hmac
import re
from typing import Optional

from fastapi import Header, Request


class ApiError(Exception):
    """One error shape for the whole API. Never carries tracebacks, paths or employee data."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class ModelUnavailable(ApiError):
    def __init__(self, name: str, reason: str = ""):
        msg = f"The '{name}' capability is not available on this server."
        super().__init__(503, "model_unavailable", f"{msg} {reason}".strip())


def _match(candidate: str, keys: tuple) -> bool:
    ok = False
    for k in keys:                                   # constant-time comparison, no early exit
        ok |= hmac.compare_digest(candidate.encode(), k.encode())
    return ok


async def require_api_key(request: Request, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    s = request.app.state.settings
    if not s.require_auth:
        return
    if not s.api_keys:                               # a misconfiguration must never silently open the API
        raise ApiError(503, "auth_not_configured", "Authentication is required but no API key is configured on the server.")
    if not x_api_key or not _match(x_api_key, s.api_keys):
        raise ApiError(401, "unauthorized", "A valid X-API-Key header is required.")


_PATH = re.compile(r"[A-Za-z]:\\[^\s'\"]+|(?:/[\w.\-]+){2,}")


def safe_text(msg: object, limit: int = 220) -> str:
    """Strip filesystem paths from a message and cap its length (used for health details)."""
    return _PATH.sub("<path>", str(msg))[:limit]
