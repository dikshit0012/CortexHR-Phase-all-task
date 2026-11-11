"""HTTP layer for Task 10:  POST /v1/chat/converse   (header: X-Demo-User, optional;  body: {"message", "session_id"?}).
The existing Task 9 `POST /v1/chat` is untouched. Identity comes from the header + the server-side demo-user table, never from the body."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

# pyrefly: ignore[missing-import]
from fastapi import APIRouter, Header, HTTPException, Request
# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field

from .classifier import IntentClassifier, Task9Classifier
from .config import Settings
from .dialogue import MAX_CHARS, DialogueManager
from .identity import DemoUserDirectory, IdentityError, resolve_identity
from .people import EmployeeDirectory
from .policy import load_policies
from .service_client import AiServiceClient, HRServiceClient
from .sessions import SessionIdError, SessionStore

log = logging.getLogger("cortexhr.task10.api")
router = APIRouter(prefix="/v1")


class ConverseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(max_length=MAX_CHARS)
    session_id: Optional[str] = Field(default=None, max_length=64, description="Omit on the first message; send the returned id afterwards.")


class ConverseResponse(BaseModel):
    session_id: str
    new_session: bool
    intent: Optional[str] = None
    confidence: Optional[float] = None
    reply: str
    fallback: bool
    action: str
    needs: Optional[str] = None
    slots: Dict[str, str] = {}
    followup: bool = False


def build_manager(settings: Settings, classifier: Optional[IntentClassifier] = None, *, directory: Optional[EmployeeDirectory] = None,
                  hr_client: Optional[HRServiceClient] = None, ai_client: Optional[AiServiceClient] = None) -> DialogueManager:
    return DialogueManager(
        classifier=classifier or Task9Classifier.load(), policies=load_policies(settings.responses_csv),
        hr=hr_client or HRServiceClient(settings.hr_base_url, connect_timeout=settings.connect_timeout, read_timeout=settings.read_timeout),
        ai=ai_client or AiServiceClient(settings.ai_base_url, connect_timeout=settings.connect_timeout, read_timeout=settings.read_timeout),
        sessions=SessionStore(ttl_seconds=settings.session_ttl_seconds), today=settings.today, directory=directory)


def attach(app, settings: Settings, **kw: Any) -> None:
    """Call from the app factory / lifespan: app.state.converse (manager) and app.state.demo_users (identity table).
    If anything fails to load, only /v1/chat/converse answers 503; every other endpoint keeps working."""
    try:
        app.state.converse = build_manager(settings, **kw)
        app.state.demo_users = DemoUserDirectory.from_csv(settings.demo_users_csv) if settings.demo_users_csv else DemoUserDirectory({})
    except Exception as exc:                                         # noqa: BLE001
        log.error("task 10 converse unavailable (%s)", type(exc).__name__)
        app.state.converse, app.state.demo_users = None, None


@router.post("/chat/converse", response_model=ConverseResponse, tags=["chat"],
             responses={401: {"description": "X-Demo-User is malformed or unknown."}, 503: {"description": "Task 10 chat is not loaded."}})
def converse(body: ConverseRequest, request: Request, x_demo_user: Optional[str] = Header(default=None)) -> ConverseResponse:
    manager = getattr(request.app.state, "converse", None)
    users = getattr(request.app.state, "demo_users", None)
    if manager is None or users is None:
        raise HTTPException(status_code=503, detail="Chat is unavailable.")
    try:
        identity = resolve_identity(x_demo_user, users)
    except IdentityError:
        raise HTTPException(status_code=401, detail="Invalid or unknown X-Demo-User.")          # never echoes the supplied value
    try:
        r = manager.handle(identity, body.session_id, body.message)
    except SessionIdError:
        raise HTTPException(status_code=422, detail="session_id must be 16-64 characters: letters, digits, '_' or '-'.")
    return ConverseResponse(session_id=r.session_id, new_session=r.new_session, intent=r.intent, confidence=r.confidence, reply=r.reply,
                            fallback=r.fallback, action=r.action, needs=r.needs, slots=r.slots, followup=r.followup)
