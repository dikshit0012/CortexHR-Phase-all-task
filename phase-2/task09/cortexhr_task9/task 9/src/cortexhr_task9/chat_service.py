"""Chat orchestration: classify -> decide (confidence, scope, approved answer) -> reply.  Holds a model and an FAQ responder that
were loaded ONCE; calling `reply` never trains, reloads or reads files.

Fallback triggers (all return fallback=True):
    empty_message        empty / whitespace-only message                        intent None, confidence None
    inference_error      the model failed or returned an invalid distribution   intent None, confidence None (details only in the log)
    low_confidence       highest probability < threshold                        intent None, confidence = the value
    out_of_scope_intent  the model says the message is not an HR question       intent 'out_of_scope'
    no_approved_answer   intent is recognised but hr_faq has no approved answer  intent = predicted intent
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from .faq import FaqResponder, load_responder
from .intent_model import LoadedIntentModel, load_artifact
from .paths import DEFAULT_FAQ, DEFAULT_MODEL, ENV_FAQ, ENV_MODEL, ENV_THRESHOLD, INTENT_FAQ_MAP, env_path
from .text_cleaning import clean_text

log = logging.getLogger("cortexhr.chat")

NO_ANSWER_FALLBACK = "I don't have an approved answer for that question yet. Please contact HR for assistance."
GENERIC_FALLBACK = "I'm not sure I understood your question. Please rephrase it, or contact HR for assistance."
OUT_OF_SCOPE_INTENT = "out_of_scope"
ANSWERED, EMPTY, ERROR, LOW_CONFIDENCE, OUT_OF_SCOPE, NO_ANSWER = (
    "answered", "empty_message", "inference_error", "low_confidence", "out_of_scope_intent", "no_approved_answer")


@dataclass(frozen=True)
class ChatResult:
    intent: Optional[str]
    confidence: Optional[float]
    reply: str
    fallback: bool
    reason: str
    faq_ids: Tuple[str, ...] = ()


class ChatService:
    def __init__(self, model: LoadedIntentModel, responder: FaqResponder, threshold: Optional[float] = None):
        thr = model.threshold if threshold is None else threshold
        if not 0.0 < thr < 1.0:
            raise ValueError("confidence threshold must be between 0 and 1 (exclusive)")
        self.model, self.responder, self.threshold = model, responder, float(thr)

    def info(self) -> dict:
        """Safe metadata: versions and counts only (no paths, no training examples, no FAQ text)."""
        m = self.model.metadata
        return {"model_version": self.model.version, "intents": len(self.model.classes), "threshold": self.threshold,
                "trained_rows": m["training"]["rows"], "sklearn_version": m["sklearn_version"], "created_at": m["created_at"]}

    def reply(self, message: str) -> ChatResult:
        cleaned = clean_text(message)
        if not cleaned:
            return ChatResult(None, None, GENERIC_FALLBACK, True, EMPTY)
        try:
            pred = self.model.predict(message)
            confidence = round(pred.confidence, 4)
        except Exception as exc:                         # noqa: BLE001 - never leak internals; the type alone is logged
            log.error("intent prediction failed (%s)", type(exc).__name__)
            return ChatResult(None, None, GENERIC_FALLBACK, True, ERROR)

        if pred.confidence < self.threshold:
            return ChatResult(None, confidence, GENERIC_FALLBACK, True, LOW_CONFIDENCE)
        if pred.intent == OUT_OF_SCOPE_INTENT:
            return ChatResult(pred.intent, confidence, GENERIC_FALLBACK, True, OUT_OF_SCOPE)

        try:
            answer = self.responder.answer(pred.intent, pred.cleaned_text)
            if answer is not None:
                return ChatResult(pred.intent, confidence, answer.reply, False, ANSWERED, answer.faq_ids)
            contact = self.responder.contact_note(pred.intent)
        except Exception as exc:                         # noqa: BLE001
            log.error("FAQ lookup failed (%s)", type(exc).__name__)
            return ChatResult(None, None, GENERIC_FALLBACK, True, ERROR)
        reply = NO_ANSWER_FALLBACK + (f" {contact}" if contact else "")
        return ChatResult(pred.intent, confidence, reply, True, NO_ANSWER)


def load_chat_service(environ: Optional[Mapping[str, str]] = None) -> ChatService:
    """Load the artifact, FAQ and mapping once. Any problem raises ModelLoadError / FaqError (clear message, no paths)."""
    env = os.environ if environ is None else environ
    model = load_artifact(env_path(ENV_MODEL, DEFAULT_MODEL, env))
    responder = load_responder(env_path(ENV_FAQ, DEFAULT_FAQ, env), INTENT_FAQ_MAP, model.classes)
    raw = env.get(ENV_THRESHOLD)
    threshold = None
    if raw:
        try:
            threshold = float(raw)
        except ValueError:
            raise ValueError(f"{ENV_THRESHOLD} must be a number between 0 and 1, got {raw!r}") from None
    return ChatService(model, responder, threshold)
