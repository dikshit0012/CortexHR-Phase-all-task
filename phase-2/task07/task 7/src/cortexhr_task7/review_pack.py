"""Task 7 - the 20-row evidence sample and the human-readability review sheet.

The sample contains ACTUAL flagged records only. Selection is deterministic and independent of row order:
  1. the best-ranked flag of every reason code that occurs (so every kind of explanation is shown), then
  2. the next best-ranked flags with an estimated amount until `n` rows are reached.
If fewer than `n` flags exist, all of them are returned (rows are never invented or repeated).
The review sheet never contains employee ids. Human columns are left EMPTY: feedback must come from a real person.
"""
from __future__ import annotations

from typing import List

import pandas as pd

from .reason_codes import PRIORITY
from .reason_text import BANNED_TERMS

HUMAN_COLUMNS = ["clear_what_looks_unusual (Y/N)", "clear_why_it_matters (Y/N)", "clear_what_to_check_next (Y/N)",
                 "free_of_technical_words (Y/N)", "suggested_rewording", "reviewer_comments"]


def select_sample(official: pd.DataFrame, n: int = 20) -> List[str]:
    """result_ids of the sample, in ranking order."""
    official = official.sort_values("result_id", kind="mergesort")           # ranking order, whatever order the rows arrive in
    if len(official) <= n:
        return list(official["result_id"])
    first_of_code = (official.groupby("reason_code", sort=False)["result_id"].min())
    ordered = [first_of_code[c] for c in PRIORITY if c in first_of_code.index]
    chosen = list(dict.fromkeys(ordered))[:n]
    if len(chosen) < n:
        pool = official[official["money_at_risk"].notna() & ~official["result_id"].isin(chosen)]
        chosen += list(pool["result_id"])[: n - len(chosen)]
    if len(chosen) < n:                                       # still short: fill with next ranked rows of any kind
        rest = official[~official["result_id"].isin(chosen)]
        chosen += list(rest["result_id"])[: n - len(chosen)]
    return sorted(chosen)


def lint_text(text: str) -> str:
    """Automated pre-check of one sentence group (NOT a substitute for a human reading it)."""
    low = text.lower()
    if any(term in low for term in BANNED_TERMS):
        return "FAIL: technical word"
    if "please" not in low:
        return "FAIL: no next step"
    if len(text.split()) > 70:
        return "FAIL: too long"
    return "PASS"


def build_review_sheet(sample: pd.DataFrame) -> pd.DataFrame:
    """One row per sample flag for a human reader; no employee ids; human columns blank."""
    sheet = pd.DataFrame({
        "review_no": range(1, len(sample) + 1),
        "result_id": sample["result_id"].to_numpy(),
        "record_id": sample["record_id"].to_numpy(),
        "period": sample["period"].to_numpy(),
        "reason_code": sample["reason_code"].to_numpy(),
        "money_at_risk": sample["money_at_risk"].to_numpy(),
        "reason_text": sample["reason_text"].to_numpy(),
        "automated_lint (not human feedback)": [lint_text(t) for t in sample["reason_text"]],
    })
    for col in HUMAN_COLUMNS:
        sheet[col] = ""
    return sheet
