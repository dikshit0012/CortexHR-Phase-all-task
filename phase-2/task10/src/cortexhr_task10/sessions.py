"""In-memory session store. Keyed by (owner, session_id) so two users who send the same id get two separate sessions.
State: previous intent, month/department/role slots and one pending clarification. NEVER stored: message text, employee names,
payslip / salary values. Limits: lost on restart, not shared between worker processes (documented in the README)."""
from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")        # long, opaque: an employee id ('EMP-0188') or e-mail cannot match


class SessionIdError(ValueError):
    pass


def validate_session_id(value: str) -> str:
    if not isinstance(value, str) or not SESSION_ID_RE.match(value):
        raise SessionIdError("session_id must be 16-64 characters: letters, digits, '_' or '-'")
    return value


@dataclass
class Pending:
    intent: str
    slot: str


@dataclass
class SessionState:
    last_intent: Optional[str] = None
    slots: Dict[str, str] = field(default_factory=dict)       # 'month' | 'department' | 'role'
    pending: Optional[Pending] = None
    turns: int = 0

    def clear_context(self) -> None:
        """Forget intent, slots and any pending question (used after a refusal and on a change of topic)."""
        self.last_intent, self.slots, self.pending = None, {}, None


class SessionStore:
    def __init__(self, ttl_seconds: float = 1800.0, max_sessions: int = 5000, clock: Callable[[], float] = time.monotonic):
        self.ttl, self.max_sessions, self._clock = ttl_seconds, max_sessions, clock
        self._data: Dict[Tuple[str, str], Tuple[SessionState, float]] = {}
        self._lock = threading.Lock()

    def get_or_create(self, owner: Optional[str], session_id: Optional[str]) -> Tuple[SessionState, str, bool]:
        """Returns (state, session_id_in_use, is_new). A missing, unknown, expired or someone else's id yields a FRESH session
        under a new random id - the caller can never read another owner's state, and a session id proves nothing."""
        now, owner_key = self._clock(), owner or "anonymous"
        with self._lock:
            self._purge(now)
            if session_id is not None:
                validate_session_id(session_id)
                hit = self._data.get((owner_key, session_id))
                if hit is not None:
                    self._data[(owner_key, session_id)] = (hit[0], now)
                    return hit[0], session_id, False
            new_id = secrets.token_urlsafe(24)
            if len(self._data) >= self.max_sessions:
                oldest = min(self._data, key=lambda k: self._data[k][1]); del self._data[oldest]
            state = SessionState()
            self._data[(owner_key, new_id)] = (state, now)
            return state, new_id, True

    def _purge(self, now: float) -> None:
        for k in [k for k, (_, t) in self._data.items() if now - t > self.ttl]:
            del self._data[k]

    def __len__(self) -> int:
        return len(self._data)
