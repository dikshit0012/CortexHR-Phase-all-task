"""Who is asking, and in which role.  The identity comes ONLY from the request context (the `X-Demo-User` header of the chat
request, the same demo mechanism the mock HR service uses) and the role ONLY from a server-side demo-user table
(`mock_data/demo_users.csv`: employee_id, role). Nothing in the message text, a slot or a session can set either.

DEMO MECHANISM, NOT AUTHENTICATION: the mock service (and therefore this chat API in demo mode) accepts whichever employee id the
client puts in the header. Anyone who can reach the API can claim any demo user. A real deployment needs a verified token."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

KNOWN_ROLES = frozenset({"EMPLOYEE", "HR", "ADMIN"})        # the roles named in chatbot_responses.allowed_roles
EMPLOYEE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")      # also keeps path/header injection out of every service call


class IdentityError(Exception):
    """Header present but unusable (bad format or not a known demo user)."""


@dataclass(frozen=True)
class Identity:
    employee_id: Optional[str]
    role: Optional[str]                                      # upper-case, one of KNOWN_ROLES, or None when unknown

    @property
    def is_anonymous(self) -> bool:
        return self.employee_id is None


ANONYMOUS = Identity(None, None)


class DemoUserDirectory:
    def __init__(self, users: Mapping[str, str]):
        self._users: Dict[str, str] = {k: str(v).strip().upper() for k, v in users.items()}

    @classmethod
    def from_csv(cls, path: Path) -> "DemoUserDirectory":
        with open(path, newline="", encoding="utf-8") as fh:
            return cls({r["employee_id"]: r["role"] for r in csv.DictReader(fh)})

    def lookup(self, employee_id: str) -> Optional[Identity]:
        role = self._users.get(employee_id)
        if role is None:
            return None
        return Identity(employee_id, role if role in KNOWN_ROLES else None)     # an unrecognised role string stays 'unknown'


def resolve_identity(header_value: Optional[str], users: DemoUserDirectory) -> Identity:
    """No header -> anonymous (only public intents can be answered). A header that is malformed or names nobody -> IdentityError."""
    if header_value is None or header_value.strip() == "":
        return ANONYMOUS
    value = header_value.strip()
    if not EMPLOYEE_ID_RE.match(value):
        raise IdentityError("invalid identity")
    found = users.lookup(value)
    if found is None:
        raise IdentityError("unknown identity")
    return found
