"""Month extraction. Output is always 'YYYY-MM' (the format the mock HR service requires for payslips and attendance).

Rules (all deterministic - the reference date is passed in, never read from the clock here):
  * 'March 2026', '2026-03', '03/2026'          -> explicit year, used as given (a month in the future is rejected)
  * 'March' (no year)                           -> the most recent such month that is NOT after the reference month.
                                                   The resolved 'YYYY-MM' is echoed back to the user in every reply.
  * 'last/previous/prior month', 'this/current month' -> relative to the reference month
  * 'next month', two different months in one message, or a month in the future -> status 'invalid' (the bot asks again)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional, Set, Tuple

MONTH_NAMES = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
               "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
               "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12}
_NAME_ALT = "|".join(sorted(MONTH_NAMES, key=len, reverse=True))
_NAME = re.compile(rf"\b({_NAME_ALT})\b(?:\.?,?\s+(20\d\d)\b)?", re.I)
_ISO = re.compile(r"\b(20\d\d)-(0?[1-9]|1[0-2])\b")
_SLASH = re.compile(r"\b(0?[1-9]|1[0-2])/(20\d\d)\b")
_PREV = re.compile(r"\b(?:last|previous|prior|past)\s+month\b", re.I)
_CUR = re.compile(r"\b(?:this|current)\s+month\b", re.I)
_NEXT = re.compile(r"\bnext\s+month\b", re.I)
# 'may' is also an ordinary word, so it only counts as a month with a month-like context
_MAY_CTX = re.compile(r"\b(?:for|in|of|during|from|since|until|till|and|about|than|to)\s+$", re.I)
MONTH_WORDS = set(MONTH_NAMES) | {"last", "previous", "prior", "past", "this", "current", "next", "month", "year"}


@dataclass(frozen=True)
class MonthResult:
    status: str                     # 'none' | 'ok' | 'invalid'
    value: Optional[str] = None     # 'YYYY-MM' when ok
    reason: Optional[str] = None    # 'future' | 'multiple' | 'next_month' when invalid


def _shift(ref: date, delta: int) -> Tuple[int, int]:
    idx = ref.year * 12 + (ref.month - 1) + delta
    return idx // 12, idx % 12 + 1


def extract_month(text: str, ref: date) -> MonthResult:
    found: Set[Tuple[int, int]] = set()
    explicit_year: Set[Tuple[int, int]] = set()
    s = text or ""
    for m in _ISO.finditer(s):
        found.add((int(m.group(1)), int(m.group(2)))); explicit_year.add((int(m.group(1)), int(m.group(2))))
    for m in _SLASH.finditer(s):
        found.add((int(m.group(2)), int(m.group(1)))); explicit_year.add((int(m.group(2)), int(m.group(1))))
    for m in _NAME.finditer(s):
        name, year = m.group(1).lower(), m.group(2)
        if name == "may" and not year:
            before, after = s[:m.start()], s[m.end():]
            alone = not re.search(r"\w", before) and not re.search(r"\w", after)      # the whole message is just 'May'
            if not (alone or _MAY_CTX.search(before)):
                continue
        mon = MONTH_NAMES[name]
        if year:
            found.add((int(year), mon)); explicit_year.add((int(year), mon))
        else:
            y = ref.year if mon <= ref.month else ref.year - 1      # most recent occurrence not after the reference month
            found.add((y, mon))
    if _PREV.search(s):
        found.add(_shift(ref, -1))
    if _CUR.search(s):
        found.add((ref.year, ref.month))
    if _NEXT.search(s):
        return MonthResult("invalid", None, "next_month")
    if not found:
        return MonthResult("none")
    if len(found) > 1:
        return MonthResult("invalid", None, "multiple")
    (y, mon), = found
    if (y, mon) > (ref.year, ref.month):
        return MonthResult("invalid", None, "future")
    return MonthResult("ok", f"{y:04d}-{mon:02d}")
