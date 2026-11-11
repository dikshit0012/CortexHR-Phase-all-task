"""Intent policy (loaded from the official chatbot_responses.csv) and the ONE central authorization function.
Authorization never looks at classifier confidence, message text, or session contents: only identity + intent + target."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Optional

from .identity import KNOWN_ROLES, Identity

SALARY_OF_OTHER = "salary_of_other"                          # official intent name (chatbot_responses.intent)
REFUSE_MARKER = "REFUSE"                                     # official backend_or_service_call value for that intent
PAY_INTENTS = frozenset({"payslip_status", "salary_band_query", "payroll_flags", SALARY_OF_OTHER})


@dataclass(frozen=True)
class IntentPolicy:
    intent: str
    template: str
    requires_login: bool
    allowed_roles: Optional[FrozenSet[str]]                  # None means ALL
    call: Optional[str]
    on_failure: Optional[str]


def load_policies(path: Path) -> Dict[str, IntentPolicy]:
    out: Dict[str, IntentPolicy] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            roles = r["allowed_roles"].strip().upper()
            allowed = None if roles == "ALL" else frozenset(x.strip() for x in roles.split(",") if x.strip())
            out[r["intent"]] = IntentPolicy(r["intent"], r["response_template"], r["requires_login"].strip().upper() == "Y",
                                            allowed, (r["backend_or_service_call"] or "").strip() or None,
                                            (r["on_failure"] or "").strip() or None)
    return out


def placeholders(text: Optional[str]) -> set:
    return set(re.findall(r"\{(\w+)\}", text or ""))


@dataclass(frozen=True)
class Decision:
    allowed: bool
    code: str            # 'ok' | 'unknown_intent' | 'refuse_always' | 'other_person' | 'login_required' | 'unknown_role' | 'role_not_allowed'


def authorize(identity: Identity, intent: Optional[str], policies: Dict[str, IntentPolicy], *, target_is_self: bool = True) -> Decision:
    """Decide BEFORE any service call.  Order matters: the salary rule is checked first and cannot be overridden by anything."""
    if intent == SALARY_OF_OTHER:
        return Decision(False, "refuse_always")
    policy = policies.get(intent or "")
    if policy is None:
        return Decision(False, "unknown_intent")
    if policy.call == REFUSE_MARKER:
        return Decision(False, "refuse_always")
    if not policy.requires_login:
        return Decision(True, "ok") if target_is_self else Decision(False, "other_person")
    if not target_is_self:
        return Decision(False, "other_person")
    if identity.is_anonymous:
        return Decision(False, "login_required")
    if identity.role not in KNOWN_ROLES:
        return Decision(False, "unknown_role")
    if policy.allowed_roles is not None and identity.role not in policy.allowed_roles:
        return Decision(False, "role_not_allowed")
    return Decision(True, "ok")
