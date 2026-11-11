"""Settings. The reference date for 'last month' / 'this month' is explicit: set CORTEXHR_REFERENCE_DATE=YYYY-MM-DD to pin it
(tests always do); when unset, today's date is used."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

TASK10_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    hr_base_url: str = "http://127.0.0.1:9000"
    ai_base_url: Optional[str] = None
    reference_date: Optional[date] = None
    session_ttl_seconds: float = 1800.0
    connect_timeout: float = 2.0
    read_timeout: float = 5.0
    responses_csv: Path = TASK10_ROOT / "data" / "chatbot_responses.csv"
    demo_users_csv: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ
        ref = e.get("CORTEXHR_REFERENCE_DATE")
        users = e.get("CORTEXHR_DEMO_USERS_CSV")
        return cls(hr_base_url=e.get("CORTEXHR_HR_BASE_URL", cls.hr_base_url), ai_base_url=e.get("CORTEXHR_AI_BASE_URL") or None,
                   reference_date=date.fromisoformat(ref) if ref else None,
                   session_ttl_seconds=float(e.get("CORTEXHR_SESSION_TTL", cls.session_ttl_seconds)),
                   demo_users_csv=Path(users) if users else None)

    def today(self) -> date:
        return self.reference_date or date.today()
