"""Department / role / leave-type vocabulary, follow-up cues and the 'is this message ONLY a slot value?' test.

Vocabulary source: the entity values in the Task 9 training data (`chatbot_nlu_training`: 7 departments, 7 roles) plus the
alias 'Support' -> 'Customer Support' (used by official dialogue DLG-10). Unknown words are never guessed into a slot."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .months import MONTH_WORDS, extract_month

DEPARTMENTS = ["Customer Support", "Data & AI", "Engineering", "Finance", "Marketing", "Operations", "Sales"]
DEPARTMENT_ALIASES = {"support": "Customer Support", "data and ai": "Data & AI", "data & ai": "Data & AI", "data ai": "Data & AI"}
ROLES = ["Account Executive", "Accountant", "Data Analyst", "HR Executive", "Product Manager", "Software Engineer", "Support Executive"]
LEAVE_TYPE_WORDS = {"casual", "sick", "earned", "privilege", "cl", "sl", "el", "pl", "leave", "leaves", "balance"}


def _phrases(table: Dict[str, str]) -> List[Tuple[str, str]]:
    return sorted(table.items(), key=lambda kv: -len(kv[0]))


_ROLE_MAP = {r.lower(): r for r in ROLES}
_DEPT_MAP = {d.lower(): d for d in DEPARTMENTS}
_DEPT_MAP.update({k: v for k, v in DEPARTMENT_ALIASES.items()})


def _find(text: str, table: Dict[str, str], blocked: List[Tuple[int, int]]) -> List[Tuple[str, int, int]]:
    out = []
    low = (text or "").lower()
    for phrase, canon in _phrases(table):
        for m in re.finditer(rf"(?<![\w&]){re.escape(phrase)}(?![\w&])", low):
            span = (m.start(), m.end())
            if any(not (span[1] <= b0 or span[0] >= b1) for b0, b1 in blocked):
                continue
            out.append((canon, *span)); blocked.append(span)
    return out


def extract_entities(text: str) -> Dict[str, object]:
    """Returns {'roles': [...], 'departments': [...], 'spans': [...]} (roles are matched first so that 'Support Executive'
    is a role and not the department 'Support')."""
    blocked: List[Tuple[int, int]] = []
    roles = _find(text, _ROLE_MAP, blocked)
    depts = _find(text, _DEPT_MAP, blocked)
    return {"roles": sorted({r[0] for r in roles}), "departments": sorted({d[0] for d in depts}), "spans": list(blocked)}


def extract_role(text: str) -> Optional[str]:
    r = extract_entities(text)["roles"]
    return r[0] if len(r) == 1 else None


def extract_department(text: str) -> Optional[str]:
    d = extract_entities(text)["departments"]
    return d[0] if len(d) == 1 else None


FOLLOWUP_CUE = re.compile(r"^\W*(?:and|also|then|what about|how about|what of|same for|now)\b", re.I)
DETAIL_CUE = re.compile(r"^\W*(?:(?:show|give)(?: me)?\s+)?(?:more\s+)?(?:details?|breakdown|reasons?)\W*$|^\W*why\b.{0,30}$", re.I)
_STOP = {"and", "also", "then", "what", "about", "how", "for", "in", "on", "of", "the", "a", "an", "only", "is", "it", "its",
         "please", "pls", "ok", "okay", "same", "too", "just", "now", "do", "me", "my", "this", "that", "show", "see", "can",
         "you", "tell", "give", "want", "would", "like", "as", "well", "to", "i", "have", "has", "left", "year"}


def has_followup_cue(text: str) -> bool:
    return bool(FOLLOWUP_CUE.match(text or "") or DETAIL_CUE.match(text or ""))


def slot_only(text: str, ref) -> bool:
    """True when the message contains a slot value (month / department / role / leave type) and NOTHING else of substance,
    e.g. 'and last month?', 'what about April?', 'and in Engineering?', 'for Support only', 'and sick leave?'.
    An unknown leftover word (such as a person's name) makes this False on purpose: that message is never treated as a follow-up."""
    ents = extract_entities(text)
    low = (text or "").lower()
    has_slot = bool(ents["roles"] or ents["departments"]) or extract_month(text, ref).status != "none"
    has_leave = bool(re.search(r"\b(casual|sick|earned|privilege|cl|sl|el)\b", low))
    if not (has_slot or has_leave):
        return False
    for s, e in sorted(ents["spans"], reverse=True):
        low = low[:s] + " " + low[e:]
    words = re.findall(r"[a-z]+", low)
    leftover = [w for w in words if w not in _STOP and w not in MONTH_WORDS and w not in LEAVE_TYPE_WORDS]
    return not leftover
