from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ...registry import CHATBOT
from ...schemas import ChatRequest, ChatResponse, ErrorResponse
from ...security import require_api_key

router = APIRouter(tags=["HR Chatbot"], dependencies=[Depends(require_api_key)],
                   responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})


@router.post("/chat", response_model=ChatResponse, summary="Ask the HR FAQ chatbot",
             openapi_extra={"requestBody": {"content": {"application/json": {"example": {"message": "What is the notice period?", "session_id": "demo-1"}}}}})
def chat(body: ChatRequest, request: Request):
    """Task 9 intent classifier + approved FAQ answers + confidence-based fallback. The chatbot answers **only from the
    approved FAQ list**; it has no access to employee records, so it cannot reveal anyone's salary. Questions about another
    person's pay are classified as such and answered with the standard fallback. No conversation state is stored on the server."""
    svc = request.app.state.registry.get(CHATBOT)
    r = svc.reply(body.message)
    return ChatResponse(reply=r.reply, intent=r.intent, confidence=r.confidence, fallback=bool(r.fallback),
                        fallback_reason=r.reason if r.fallback else None, faq_ids=list(r.faq_ids), session_id=body.session_id,
                        model_version=svc.info()["model_version"], request_id=request.state.request_id)
