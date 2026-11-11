"""Runs the OFFICIAL dialogues (data/chatbot_dialogues.csv, read-only) turn by turn and records, per turn, what happened and
whether each derived expectation held. Used by test_official_dialogues.py and by scripts/run_evidence.py, so the evidence files and
the tests can never disagree.

Official expectations = the `expected_intent` column. Everything else is DERIVED from the official chatbot_responses.csv:
  * allowed role      -> authorization must be 'ok'  (salary_of_other: 'refuse_always')
  * service call      -> exactly one call whose method+path starts with the CSV value; no call for faq/None/REFUSE intents
  * follow-up rows    -> context_note 'follow-up: reuse entity/slot' must be resolved as a follow-up with the same intent
Slot expectations below are derived from the reference date (2026-04-15) and the wording of each turn."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cortexhr_task10.config import TASK10_ROOT
from cortexhr_task10.policy import load_policies
from support import RESPONSES

DIALOGUES = TASK10_ROOT / "data" / "chatbot_dialogues.csv"
ROLE_USER = {"employee": "EMP-1001", "hr": "EMP-2001"}
EXPECTED_SLOTS: Dict[Tuple[str, int], Dict[str, str]] = {
    ("DLG-02", 1): {"month": "2026-03"}, ("DLG-04", 1): {"month": "2026-03"}, ("DLG-05", 2): {"month": "2026-03"},
    ("DLG-07", 1): {"department": "Sales"}, ("DLG-07", 2): {"department": "Engineering"}, ("DLG-07", 3): {"department": "Engineering"},
    ("DLG-08", 1): {"month": "2026-04"}, ("DLG-08", 2): {"month": "2026-04"},
    ("DLG-09", 1): {"role": "Data Analyst"}, ("DLG-09", 2): {"role": "Software Engineer"}, ("DLG-11", 1): {"role": "Data Analyst"}}


def load_dialogues() -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    with open(DIALOGUES, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["dialogue_id"], []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: int(r["turn_no"]))
    return out


def call_prefix(call: Optional[str]) -> Optional[str]:
    if not call or call == "REFUSE" or call.startswith("faq:"):
        return None
    m = re.match(r"^(GET|POST)\s+(/[^\s?{(]*)", call)
    return f"{m.group(1)} {m.group(2)}" if m else None


@dataclass
class TurnRecord:
    dialogue_id: str
    turn: int
    user_role: str
    text: str
    expected_intent: str
    context_note: str
    actual_intent: Optional[str]
    confidence: Optional[float]
    action: str
    authorization: str
    slots: Dict[str, str]
    followup: bool
    reply: str
    service_calls: Tuple[str, ...]
    observed_http: List[str]
    checks: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def run_dialogue(env, dialogue_id: str, rows: List[dict]) -> List[TurnRecord]:
    policies = load_policies(RESPONSES)
    emp, sid, records, first_reply = ROLE_USER[rows[0]["user_role"]], None, [], None
    for r in rows:
        h0, a0 = len(env.hr.log), len(env.ai.log)
        res = env.say(emp, r["text"], sid); sid = res.session_id
        observed = [f"{m} {p}" for m, p, _ in env.hr.log[h0:]] + [f"{m} {p}" for m, p, _ in env.ai.log[a0:]]
        pol, exp = policies[r["expected_intent"]], r["expected_intent"]
        prefix, turn = call_prefix(pol.call), int(r["turn_no"])
        rec = TurnRecord(dialogue_id, turn, r["user_role"], r["text"], exp, (r.get("context_note") or "").strip(), res.intent, res.confidence, res.action,
                         res.authorization, dict(res.slots), res.followup, res.reply, res.service_calls, observed)
        c = rec.checks
        c["intent == official expected_intent"] = res.intent == exp
        c["authorization as configured"] = res.authorization == ("refuse_always" if exp == "salary_of_other" else "ok")
        if prefix is None:
            c["no service call (none configured / refused)"] = observed == [] and res.service_calls == ()
        else:
            c["exactly one service call, matching the configured endpoint"] = len(observed) == 1 and observed[0].startswith(prefix.split(" ", 1)[0]) \
                and prefix.split(" ", 1)[1] in observed[0] and len(res.service_calls) == 1
            c["answered (template filled, no failure)"] = res.action == "answered"
        if exp == "salary_of_other":
            c["refused with the official response_template"] = res.action == "refused" and res.reply == pol.template
        if rec.context_note.startswith("follow-up"):
            c["resolved as a follow-up (context reused)"] = res.followup
        if (dialogue_id, turn) in EXPECTED_SLOTS:
            want = EXPECTED_SLOTS[(dialogue_id, turn)]
            c[f"slots contain {want}"] = all(res.slots.get(k) == v for k, v in want.items())
        if dialogue_id == "DLG-01" and turn == 1:
            first_reply = res.reply
        if dialogue_id == "DLG-01" and turn == 2:
            c["follow-up reuses the earlier intent and answers the same question"] = res.reply == first_reply
        if res.action == "failed":
            rec.notes.append("service call failed")
        records.append(rec)
    return records
