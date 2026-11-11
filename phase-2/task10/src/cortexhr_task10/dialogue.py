"""Dialogue manager: classification -> pre-authorization checks -> follow-up resolution -> authorization -> slot filling ->
service call -> reply. Each step is a separate module; this file only orchestrates.

Safety invariants (each one has tests):
  * Requests about another person's pay are refused BEFORE authorization and BEFORE any service call, whatever the classifier,
    the session or the caller's role says. A refusal also wipes the session context, so 'and last month?' cannot revive it.
  * Authorization is re-run on every turn (including follow-ups) with the CURRENT identity; a session proves nothing.
  * Service clients only ever receive the authenticated employee id; there is no code path that accepts another one.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Dict, Optional, Tuple

from .classifier import IntentClassifier, Prediction
from .identity import Identity
from .months import MonthResult, extract_month
from .people import EmployeeDirectory
from .policy import PAY_INTENTS, SALARY_OF_OTHER, IntentPolicy, authorize
from .render import render
from .sensitive import detect_other_reference, has_pay_term
from .service_client import AiServiceClient, HRServiceClient, ServiceError
from .sessions import Pending, SessionState, SessionStore
from .slots import DETAIL_CUE, extract_department, extract_role, has_followup_cue, slot_only

log = logging.getLogger("cortexhr.task10.dialogue")
MAX_CHARS = 1000
GENERIC_FALLBACK = "I'm not sure I understood your question. Please rephrase it, or contact HR for assistance."
OWN_DATA_ONLY = "I can only share your own information. Please contact HR if you need help with an authorized request."
SALARY_REFUSAL_DEFAULT = "Sorry, I cannot share another person's pay. You can ask about your own payslip."
LOGIN_REQUIRED = "Please sign in to ask about that."
NOT_ALLOWED = "Sorry, you don't have access to that information."
ASK = {"month": 'Which month would you like? (for example "March 2026" or "last month")',
       "department": "Which department do you mean?", "role": "Which role do you mean?"}
# Exact one/two-word greetings the Task 9 model is unreliable on (e.g. 'hi' scored 0.06). Used ONLY when the classifier is not
# confident, and it can only ever produce the static, data-free `greet` reply.
SMALLTALK = {"hi", "hii", "hello", "hey", "hiya", "yo", "good morning", "good afternoon", "good evening", "hi there", "hello there"}
KEEPS_CONTEXT = frozenset({"greet", "thanks", "goodbye", "chitchat", "help"})     # pleasantries do not end the previous topic
MONTH_INVALID = 'I couldn\'t use that month. Which month would you like? (a month that has already started, for example "March 2026" or "last month")'
ATTENDANCE_ANOMALY_FOLLOWUP = re.compile(
    r"^\W*(?:(?:are|were)\s+there\s+)?(?:any\s+)?(?:unusual|anomalous)\s+(?:attendance\s+)?days\W*$",
    re.I,
)

# Slots each intent needs. Derived from the official chatbot_responses placeholders: {month} / {dept} in the service call
# (payslip_status, attendance_summary, payroll_flags, attrition_risk_query) and the input {role} of the two POST /ai/v1 calls.
# A consistency test checks every entry against the CSV.
REQUIRED_SLOTS: Dict[str, Tuple[str, ...]] = {
    "payslip_status": ("month",), "attendance_summary": ("month",), "payroll_flags": ("month",),
    "attrition_risk_query": ("department",), "salary_band_query": ("role",), "resume_score_request": ("role",)}
SLOT_NAME = {"month": "month", "department": "dept", "role": "role"}


@dataclass(frozen=True)
class TurnResult:
    session_id: str
    new_session: bool
    intent: Optional[str]
    confidence: Optional[float]
    reply: str
    fallback: bool
    action: str                     # static | answered | clarify | refused | denied | failed | fallback
    authorization: str              # policy decision code, or 'n/a' when no decision was needed
    slots: Dict[str, str]
    followup: bool
    needs: Optional[str]
    service_calls: Tuple[str, ...]


class DialogueManager:
    def __init__(self, *, classifier: IntentClassifier, policies: Dict[str, IntentPolicy], hr: HRServiceClient, ai: AiServiceClient,
                 sessions: SessionStore, today: Callable[[], date], directory: Optional[EmployeeDirectory] = None):
        self.classifier, self.policies, self.hr, self.ai = classifier, policies, hr, ai
        self.sessions, self.today, self.directory = sessions, today, directory

    # ---------------------------------------------------------------- public
    def handle(self, identity: Identity, session_id: Optional[str], message: str) -> TurnResult:
        state, sid, new = self.sessions.get_or_create(identity.employee_id, session_id)
        state.turns += 1
        text = (message or "").strip()[:MAX_CHARS]
        ref = self.today()
        mk = lambda **kw: self._result(sid, new, state, **kw)               # noqa: E731

        if not text:
            return mk(intent=None, conf=None, reply=GENERIC_FALLBACK, action="fallback", auth="n/a", fallback=True)
        try:
            pred = self.classifier.classify(text)
        except Exception as exc:                                              # noqa: BLE001 - never leak internals
            log.error("classifier failed (%s)", type(exc).__name__)
            return mk(intent=None, conf=None, reply=GENERIC_FALLBACK, action="fallback", auth="n/a", fallback=True)
        conf = round(pred.confidence, 4)
        confident = pred.confidence >= self.classifier.threshold
        intent = pred.intent if confident else None
        if not confident and re.sub(r"[^a-z ]", "", text.lower()).strip() in SMALLTALK and "greet" in self.policies:
            intent = "greet"
        other, pay = detect_other_reference(text, self.directory), has_pay_term(text)
        protected = intent is not None and intent in self.policies and self.policies[intent].requires_login
        in_context = state.last_intent is not None and (has_followup_cue(text) or state.pending is not None or len(text.split()) <= 4)

        # 1. PRE-AUTHORIZATION: anything aimed at another person's pay is refused here, before classification can matter
        if intent == SALARY_OF_OTHER or (pay and other.any) or (pred.intent == SALARY_OF_OTHER and other.any):
            return self._refuse(state, mk, conf, salary=True)
        if other.strong and (protected or in_context):
            return self._refuse(state, mk, conf, salary=intent in PAY_INTENTS or state.last_intent in PAY_INTENTS)

        # 2. follow-up / clarification resolution
        vals = self._slot_values(text, ref)
        sl_only, cue, followup = slot_only(text, ref), has_followup_cue(text), False
        pend = state.pending
        if pend is not None:
            st, _ = vals[pend.slot]
            if st != "none" and (intent is None or intent == pend.intent or sl_only):
                intent, followup = pend.intent, True
            elif intent is not None and intent != pend.intent:
                state.pending = None                                          # the user changed topic: drop the old question
            else:
                return mk(intent=pend.intent, conf=conf, reply=ASK[pend.slot], action="clarify", auth="n/a", needs=pend.slot)
        elif state.last_intent == "attendance_summary" and ATTENDANCE_ANOMALY_FOLLOWUP.fullmatch(text):
            intent, followup = "unusual_attendance", True
        elif state.last_intent and sl_only and (cue or intent is None):
            intent, followup = state.last_intent, True
        elif intent is None and state.last_intent and DETAIL_CUE.match(text):
            intent, followup = state.last_intent, True
        elif intent is not None:
            followup = cue
        if intent is None:
            return mk(intent=None, conf=conf, reply=GENERIC_FALLBACK, action="fallback", auth="n/a", fallback=True)
        if not followup:
            state.slots = {}                                                  # no stale month/department/role for a fresh question
        state.pending = None if pend is None or intent != pend.intent else state.pending

        # 3. AUTHORIZATION (every turn, current identity, before any slot work or service call)
        decision = authorize(identity, intent, self.policies, target_is_self=True)
        if not decision.allowed:
            if decision.code == "refuse_always":
                return self._refuse(state, mk, conf, salary=True)
            state.clear_context()
            reply = {"login_required": LOGIN_REQUIRED}.get(decision.code, NOT_ALLOWED)
            return mk(intent=intent, conf=conf, reply=reply, action="denied", auth=decision.code)
        policy = self.policies[intent]

        # 4. intents that need no data
        if intent not in REQUIRED_SLOTS and (policy.call is None or policy.call.startswith("faq:")):
            state.pending = None
            if intent not in KEEPS_CONTEXT:                                   # a policy/FAQ question is a change of topic
                state.clear_context()
            return mk(intent=intent, conf=conf, reply=policy.template, action="static", auth=decision.code, followup=followup)

        # 5. slot filling (clarify instead of calling with a missing value)
        slots: Dict[str, str] = {}
        for slot in REQUIRED_SLOTS.get(intent, ()):
            st, val = vals[slot]
            if st == "ok":
                slots[slot] = val
            elif st == "invalid":
                state.last_intent, state.pending = intent, Pending(intent, slot)
                return mk(intent=intent, conf=conf, reply=MONTH_INVALID, action="clarify", auth=decision.code, needs=slot, followup=followup)
            elif followup and state.slots.get(slot):
                slots[slot] = state.slots[slot]
            else:
                state.last_intent, state.pending = intent, Pending(intent, slot)
                return mk(intent=intent, conf=conf, reply=ASK[slot], action="clarify", auth=decision.code, needs=slot, followup=followup)
        state.last_intent, state.pending, state.slots = intent, None, dict(slots)

        # 6. the (single) service call for this turn
        label = self._label(policy, slots)
        try:
            data = self._call(intent, identity.employee_id, slots, ref)
        except ServiceError as exc:
            reply = (f"I couldn't find any data for {slots['month']}. Please check the month, or contact HR."
                     if exc.kind == "not_found" and "month" in slots else (policy.on_failure or GENERIC_FALLBACK))
            return mk(intent=intent, conf=conf, reply=reply, action="failed", auth=decision.code, calls=(label,), followup=followup)
        text_out = render(intent, policy.template, data, slots, ref)
        if text_out is None:                                                   # unexpected response shape: safe failure text
            return mk(intent=intent, conf=conf, reply=policy.on_failure or GENERIC_FALLBACK, action="failed", auth=decision.code, calls=(label,), followup=followup)
        return mk(intent=intent, conf=conf, reply=text_out, action="answered", auth=decision.code, calls=(label,), followup=followup)

    # --------------------------------------------------------------- helpers
    def _refuse(self, state: SessionState, mk, conf, *, salary: bool) -> TurnResult:
        state.clear_context()
        if salary:
            tpl = self.policies.get(SALARY_OF_OTHER)
            return mk(intent=SALARY_OF_OTHER, conf=conf, reply=tpl.template if tpl else SALARY_REFUSAL_DEFAULT, action="refused", auth="refuse_always")
        return mk(intent=None, conf=conf, reply=OWN_DATA_ONLY, action="refused", auth="other_person")

    @staticmethod
    def _slot_values(text: str, ref: date) -> Dict[str, Tuple[str, Optional[str]]]:
        m: MonthResult = extract_month(text, ref)
        dept, role = extract_department(text), extract_role(text)
        return {"month": (m.status, m.value), "department": ("ok" if dept else "none", dept), "role": ("ok" if role else "none", role)}

    @staticmethod
    def _label(policy: IntentPolicy, slots: Dict[str, str]) -> str:
        call = re.sub(r"\s*\(.*?\)\s*$", "", policy.call or "")
        return re.sub(r"\{(\w+)\}", lambda m: next((slots[k] for k, v in SLOT_NAME.items() if v == m.group(1) and k in slots), m.group(0)), call)

    def _call(self, intent: str, emp: Optional[str], slots: Dict[str, str], ref: date) -> Any:
        h = self.hr
        simple = {"leave_balance": lambda: h.leave_balance(emp), "leave_request_status": lambda: h.leave_requests(emp, 1),
                  "payslip_status": lambda: h.payslip(emp, slots["month"]), "my_manager": lambda: h.employee_me(emp),
                  "my_department": lambda: h.employee_me(emp), "team_members": lambda: h.team(emp),
                  "attendance_summary": lambda: h.attendance(emp, slots["month"]), "holiday_list": lambda: h.holidays(emp)}
        if intent in simple:
            return simple[intent]()
        a = self.ai
        ai = {"unusual_attendance": lambda: a.call("POST", "/ai/v1/anomaly/attendance", emp, json={"month": ref.strftime("%Y-%m")}),
              "salary_band_query": lambda: a.call("POST", "/ai/v1/salary-band", emp, json={"role": slots["role"]}),
              "attrition_risk_query": lambda: a.call("GET", "/ai/v1/attrition/risk", emp, params={"department": slots["department"]}),
              "headcount_forecast": lambda: a.call("GET", "/ai/v1/forecast/headcount", emp),
              "payroll_flags": lambda: a.call("GET", "/ai/v1/anomaly/payroll", emp, params={"month": slots["month"]}),
              "resume_score_request": lambda: a.call("POST", "/ai/v1/resume/score", emp, json={"role": slots["role"]})}
        if intent in ai:
            return ai[intent]()
        raise ServiceError("invalid")

    @staticmethod
    def _result(sid: str, new: bool, state: SessionState, *, intent, conf, reply, action, auth, fallback: bool = False,
                needs: Optional[str] = None, calls: Tuple[str, ...] = (), followup: bool = False) -> TurnResult:
        log.info("turn: intent=%s action=%s auth=%s calls=%d", intent, action, auth, len(calls))      # never the message text
        return TurnResult(sid, new, intent, conf, reply, fallback, action, auth, dict(state.slots), followup, needs, tuple(calls))
