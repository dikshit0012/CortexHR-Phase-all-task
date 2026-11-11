"""Task 7 - ranking of flags by money at risk, with a documented deterministic tie-break.

Sort keys, in order
  1. money_at_risk, largest first; records with no estimate (empty) come after every estimated one
  2. reason priority (reason_codes.PRIORITY): the more serious kind of issue first
  3. number of independent signals that fired (rules + model): more evidence first
  4. the detector's raw anomaly score, highest first (compared only within the same source, because the
     priority in step 2 already separates sources)
  5. record_id, ascending: a stable final tie-break, so row order of the input never matters

Amount, severity and confidence are different things:
  amount     = how much money could be affected (step 1);
  severity   = how serious the TYPE of issue is, a fixed business ordering, not a probability (step 2);
  confidence = how strong the evidence is (step 3: independent signals), not a probability either.
A large amount does not make an alert more likely to be a real error.
"""
from __future__ import annotations

import pandas as pd

from .reason_codes import PRIORITY_INDEX


def rank_flags(internal: pd.DataFrame) -> pd.DataFrame:
    df = internal.copy()
    unknown = sorted(set(df["primary_reason_code"]) - set(PRIORITY_INDEX))
    if unknown:
        raise ValueError(f"unknown reason code(s) cannot be ranked: {unknown}")
    df["_priority"] = df["primary_reason_code"].map(PRIORITY_INDEX)
    df = df.sort_values(
        by=["money_at_risk", "_priority", "signal_count", "anomaly_score_raw", "record_id"],
        ascending=[False, True, False, False, True], na_position="last", kind="mergesort",
    )
    return df.drop(columns="_priority").reset_index(drop=True)
