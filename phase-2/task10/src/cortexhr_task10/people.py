"""Employee-name handling.

IMPORTANT CONTRACT FACT: the mock HR service has no employee directory and no endpoint that returns another person's data (every
route is scoped to the X-Demo-User header; the `employee` query parameter is ignored). So this chatbot has no authorised use for
another person's identity. Names are therefore recognised in order to REFUSE, never to look somebody up. The directory below is an
optional input (tests and, if the operator supplies one, a deployment) that improves recognition of full / partial / duplicate names.
`disambiguation_question` is provided for a future authorised flow; the dialogue manager never uses it to reveal anything."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class EmployeeRecord:
    employee_id: str
    full_name: str
    department: Optional[str] = None      # the only 'identifying detail' offered when asking which person is meant


@dataclass(frozen=True)
class Mention:
    surface: str
    kind: str                              # 'full' | 'partial'
    matches: Tuple[EmployeeRecord, ...]

    @property
    def status(self) -> str:
        return "unique" if len(self.matches) == 1 else "ambiguous"


class EmployeeDirectory:
    def __init__(self, records: Iterable[EmployeeRecord] = ()):
        self.records: List[EmployeeRecord] = list(records)

    def find(self, text: str) -> List[Mention]:
        low = " " + re.sub(r"[^\w\s']", " ", (text or "").lower()) + " "
        out: List[Mention] = []
        covered: List[str] = []
        by_full: Dict[str, List[EmployeeRecord]] = {}
        for r in self.records:
            by_full.setdefault(r.full_name.lower(), []).append(r)
        for full, recs in by_full.items():
            if f" {full} " in low or f" {full}'s " in low:
                out.append(Mention(full.title(), "full", tuple(recs))); covered += full.split()
        parts: Dict[str, List[EmployeeRecord]] = {}
        for r in self.records:
            for tok in r.full_name.lower().split():
                parts.setdefault(tok, []).append(r)
        for tok, recs in parts.items():
            if tok in covered or len(tok) < 3:
                continue
            if re.search(rf"\b{re.escape(tok)}(?:'s)?\b", low):
                out.append(Mention(tok.title(), "partial", tuple(dict.fromkeys(recs))))
        return out


def disambiguation_question(mention: Mention) -> str:
    """Only the department is offered; never salary, payroll, contact or any other private field."""
    depts = sorted({m.department for m in mention.matches if m.department})
    if len(mention.matches) > 1:
        tail = f" (for example in {', '.join(depts)})" if depts else ""
        return f"I found more than one employee with that name. Could you clarify which one you mean{tail}?"
    return "Which employee do you mean?"
