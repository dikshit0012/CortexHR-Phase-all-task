"""Task 9 integration: POST /v1/chat (HR intent classification + approved FAQ answers + confidence fallback).

The classifier, FAQ lookup and fallback logic live in `task 9/src/cortexhr_task9`; this module only adds the HTTP layer
(request / response schemas, router, error handling) to the existing Task 8 service. It follows the Task 8 conventions:
router prefix /v1, `request.app.state` for loaded objects, 503 when something required is not loaded, 422 bodies that never
echo submitted values, and no stack traces or paths in any response.

The chat model is loaded ONCE in the lifespan handler (`load_chat_service_safely`). If it cannot be loaded, ONLY /v1/chat answers
503; the Task 8 anomaly endpoints keep working exactly as before.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ._paths import TASK8_ROOT

log = logging.getLogger("cortexhr.chat.api")
ENV_TASK9_ROOT = "CORTEXHR_TASK9_ROOT"          # folder that contains `src/cortexhr_task9`; default: discovered `task 9`
MAX_MESSAGE_CHARS = 1000


def _task9_root() -> Path:
    configured = os.environ.get(ENV_TASK9_ROOT)
    if configured:
        return Path(configured).expanduser().resolve()

    current = TASK8_ROOT
    candidates = []
    for _ in range(8):
        candidates.extend([
            current / "task 9",
            current / "task9",
            current / "cortexhr_task9" / "task 9",
            current / "task9" / "cortexhr_task9" / "task 9",
            current / "task9" / "task 9",
        ])
        current = current.parent

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if (path / "src" / "cortexhr_task9").is_dir():
            return path.resolve()
    return (TASK8_ROOT.parent / "task 9").resolve()


def _import_task9() -> Any:
    src = str(_task9_root() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from cortexhr_task9 import chat_service              # noqa: WPS433 - deliberately imported late (optional component)
    return chat_service


def load_chat_service_safely() -> Optional[Any]:
    """Return a ready ChatService, or None (logged by exception TYPE only) when the Task 9 model / FAQ cannot be loaded."""
    try:
        service = _import_task9().load_chat_service()
        log.info("chat model loaded: %s", service.model.version)
        return service
    except Exception as exc:                              # noqa: BLE001 - /v1/chat becomes 503, the rest of the app is unaffected
        name = type(exc).__name__
        detail = exc if name in ("ModelLoadError", "FaqError") else "see Task 9 README"
        log.error("chat service unavailable (%s): %s", name, detail)
        return None


class ChatRequest(BaseModel):
    """`message` is required and must be a string. Empty / whitespace-only text is valid input: it gets the safe fallback."""
    model_config = ConfigDict(extra="forbid")
    message: str = Field(max_length=MAX_MESSAGE_CHARS, description="The employee's question (max 1000 characters).")


class ChatResponse(BaseModel):
    """`intent` is null whenever the bot is uncertain or the input is unusable; `confidence` is null when no prediction exists.
    `confidence` is the classifier's top probability - it is NOT a calibrated probability of being correct."""
    intent: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    reply: str
    fallback: bool


router = APIRouter(prefix="/v1")


def get_chat_service(request: Request) -> Any:
    service = getattr(request.app.state, "chat", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Chat is unavailable: the intent model is not loaded.")
    return service


@router.post("/chat", response_model=ChatResponse, tags=["chat"],
             responses={503: {"description": "The intent model is not loaded."}})
def chat(body: ChatRequest, service: Any = Depends(get_chat_service)) -> ChatResponse:
    result = service.reply(body.message)                  # never raises: inference errors become a safe fallback inside
    log.info("chat: reason=%s fallback=%s", result.reason, result.fallback)     # no message text in logs
    return ChatResponse(intent=result.intent, confidence=result.confidence, reply=result.reply, fallback=result.fallback)
