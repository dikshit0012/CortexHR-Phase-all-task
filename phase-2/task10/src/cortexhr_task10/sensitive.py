"""Pre-authorization detection of 'this message is about somebody else' and 'this message is about pay'.
Deliberately conservative and NOT the only defence: authorization (policy.authorize) and the service client (which can only ever send
the caller's own id) hold even if every pattern here is bypassed."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .months import MONTH_NAMES
from .people import EmployeeDirectory
from .slots import DEPARTMENTS, ROLES, DEPARTMENT_ALIASES

PAY_TERMS = re.compile(r"\b(?:salary|salaries|pay|paid|payslip|payslips|payroll|ctc|earn|earns|earning|earnings|compensation|wage|wages|"
                       r"package|income|bonus|increment|hike|take[- ]home|net pay|gross|stipend|remuneration|lpa|pasylip|slaary|paylsip)\b", re.I)
_VOCAB = {w.lower() for p in (DEPARTMENTS + ROLES + list(DEPARTMENT_ALIASES)) for w in re.findall(r"[A-Za-z]+", p)}
_SAFE_CAPS = set(MONTH_NAMES) | _VOCAB | {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "hr", "form", "uan",
             "zenith", "nova", "aura", "india", "republic", "day", "diwali", "holi", "christmas", "eid", "wfh", "pf", "tds", "esi", "lop"}
_STARTERS = {"what", "who", "how", "show", "is", "tell", "can", "do", "does", "give", "please", "and", "why", "any", "my", "when",
             "where", "hi", "hello", "hey", "thanks", "thank", "which", "are", "will", "could", "would", "also", "suggest", "score",
             "forecast", "ok", "okay", "work", "salary", "payslip", "payroll", "list", "get", "need", "i", "for", "help", "bye",
             "goodbye", "yes", "no", "so", "then", "now", "that", "this", "the", "a", "an", "it", "let", "yo", "cheers"}
_STRONG_WORDS = re.compile(r"\b(?:colleagues?|co-?workers?|teammates?|someone|somebody|friends?|boss|supervisor|reportees?|subordinates?|"
                           r"his|her|hers|him|he|she|another|someone else|somebody else|other (?:employee|person|people)s?)\b", re.I)
_WEAK_WORDS = re.compile(r"\b(?:employees?|people|staff|everyone|everybody|anyone|anybody|manager|managers|lead|director|ceo|they|them|their|others?)\b", re.I)
_POSSESSIVE = re.compile(r"\b([A-Za-z]+)['\u2019]s\b")
_POSS_SAFE = {"what", "who", "that", "it", "there", "here", "how", "let", "where", "when", "which", "today", "this", "everything", "one", "tomorrow"}
_PAY_OF = re.compile(r"\b(?:salary|salaries|pay|payslip|payslips|payroll|ctc|compensation|wage|wages|package|income|bonus|increment|hike|remuneration)"
                     r"\s+(?:of|for|about)\s+(?!(?:me|my|mine|myself|the|a|an|this|last|previous|prior|current|next|\d)\b)([a-z][a-z&']*)", re.I)
_EARNS = re.compile(r"\b([a-z]+)\s+(?:earns?|makes?\s+(?:more|less)|gets?\s+paid|is\s+paid|are\s+paid|takes?\s+home)\b", re.I)
_EARNS_SKIP = {"i", "we", "you", "it", "that", "which", "who", "much", "what", "do", "does", "did", "to", "and", "or", "then", "will", "would", "can", "should"}
_MAKE = re.compile(r"\bwhat\s+(?:does|do|did|will|would)\s+(?!(?:i|you|we)\s+(?:make|get|earn))(.{1,40}?)\s+(?:make|get paid|take home|earn)\b", re.I)
_TITLE = re.compile(r"\b([A-Z][a-z]+)(?:['\u2019]s)?\b")


@dataclass(frozen=True)
class OtherRef:
    strong: bool = False            # names, possessives, pronouns, 'colleague'...: about someone else for any protected intent
    weak: bool = False              # 'employees', 'manager', 'their'...: only suspicious together with a pay term
    names: Tuple[str, ...] = ()

    @property
    def any(self) -> bool:
        return self.strong or self.weak


def has_pay_term(text: str) -> bool:
    return bool(PAY_TERMS.search(text or ""))


def _title_names(text: str) -> list:
    out = []
    toks = list(_TITLE.finditer(text))
    first_alpha = re.search(r"[A-Za-z]", text)
    first_pos = first_alpha.start() if first_alpha else 0
    for i, m in enumerate(toks):
        base = m.group(1).lower()
        if base in _SAFE_CAPS:
            continue
        if m.start() == first_pos:                      # sentence start: a name only if followed by another capitalised word
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if base in _STARTERS or nxt is None or nxt.start() - m.end() > 1 or nxt.group(1).lower() in _SAFE_CAPS | _STARTERS:
                continue
        elif base in _STARTERS:
            continue
        out.append(m.group(1))
    return out


def detect_other_reference(text: str, directory: Optional[EmployeeDirectory] = None) -> OtherRef:
    s = text or ""
    names = list(_title_names(s))
    strong = bool(names)
    for m in _POSSESSIVE.finditer(s):
        if m.group(1).lower() not in _POSS_SAFE:
            strong = True; names.append(m.group(1))
    for m in _PAY_OF.finditer(s):
        w = m.group(1).lower().rstrip("'")
        if w not in MONTH_NAMES and w not in _VOCAB:
            strong = True; names.append(m.group(1))
    for m in _EARNS.finditer(s):
        if m.group(1).lower() not in _EARNS_SKIP:
            strong = True; names.append(m.group(1))
    for m in _MAKE.finditer(s):                         # 'what does my manager make' / 'what does priya get paid'
        names.append(m.group(1).strip()); strong = True
    if _STRONG_WORDS.search(s):
        strong = True
    if directory is not None:
        found = directory.find(s)
        if found:
            strong = True; names += [f.surface for f in found]
    weak = bool(_WEAK_WORDS.search(s))
    return OtherRef(strong, weak, tuple(dict.fromkeys(names)))
