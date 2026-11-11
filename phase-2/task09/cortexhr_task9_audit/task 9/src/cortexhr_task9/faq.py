"""Approved-FAQ lookup. Completely separate from classification: it only receives an intent name and a cleaned message.

hr_faq.csv has NO intent column, so the intent -> FAQ link lives in intent_faq_map.json (hand-reviewed against the real FAQ text and
the training messages). Everything is validated at load time: unknown FAQ ids, unknown / missing intents, empty rules and
invalid regular expressions stop startup with a clear error. Answers are returned VERBATIM from hr_faq.csv; nothing is generated.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Tuple

import pandas as pd

FAQ_COLUMNS = ("faq_id", "question", "answer", "category", "policy_source")


class FaqError(RuntimeError):
    """hr_faq.csv or the intent->FAQ mapping is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class FaqEntry:
    faq_id: str
    question: str
    answer: str
    category: str
    policy_source: str


@dataclass(frozen=True)
class Rule:
    name: str
    faq_ids: Tuple[str, ...]
    require_any: Tuple[Pattern[str], ...]
    match_any: Tuple[Pattern[str], ...]
    veto_any: Tuple[Pattern[str], ...]


@dataclass(frozen=True)
class IntentEntry:
    rules: Tuple[Rule, ...]
    contact_faq_id: Optional[str]
    note: str


@dataclass(frozen=True)
class FaqAnswer:
    reply: str
    faq_ids: Tuple[str, ...]
    rule: str


def load_faq(path: Path) -> Dict[str, FaqEntry]:
    path = Path(path)
    if not path.is_file():
        raise FaqError(f"FAQ file not found: {path.name}")
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    missing = [c for c in FAQ_COLUMNS if c not in df.columns]
    if missing:
        raise FaqError(f"{path.name} is missing columns {missing}")
    entries: Dict[str, FaqEntry] = {}
    for row in df.to_dict("records"):
        fid = row["faq_id"].strip()
        if not fid or not row["answer"].strip():
            raise FaqError(f"{path.name} has a row with an empty faq_id or answer")
        if fid in entries:
            raise FaqError(f"{path.name} has a duplicate faq_id: {fid}")
        entries[fid] = FaqEntry(fid, row["question"].strip(), row["answer"].strip(), row["category"].strip(), row["policy_source"].strip())
    if not entries:
        raise FaqError(f"{path.name} contains no FAQ entries")
    return entries


def _compile(patterns: Iterable[str], where: str) -> Tuple[Pattern[str], ...]:
    out = []
    for p in patterns:
        try:
            out.append(re.compile(p))
        except re.error:
            raise FaqError(f"invalid regular expression in {where}: {p!r}") from None
    return tuple(out)


def load_mapping(path: Path, faq: Dict[str, FaqEntry], intents: Iterable[str]) -> Dict[str, IntentEntry]:
    path = Path(path)
    if not path.is_file():
        raise FaqError(f"intent->FAQ mapping not found: {path.name}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))["intents"]
    except (ValueError, KeyError):
        raise FaqError(f"{path.name} is not a valid mapping file") from None
    intents = list(intents)
    unknown = sorted(set(raw) - set(intents))
    absent = sorted(set(intents) - set(raw))
    if unknown or absent:
        raise FaqError(f"mapping and model disagree on intents (in mapping only: {unknown}; in model only: {absent})")
    result: Dict[str, IntentEntry] = {}
    for intent, spec in raw.items():
        rules: List[Rule] = []
        for r in spec.get("rules", []):
            ids = tuple(r.get("faq_ids", []))
            bad = [i for i in ids if i not in faq]
            if not ids or bad:
                raise FaqError(f"{intent}/{r.get('name')}: unknown or empty faq_ids {bad or ids}")
            if not r.get("match_any"):
                raise FaqError(f"{intent}/{r.get('name')}: a rule needs at least one match_any pattern (no catch-all rules)")
            where = f"{intent}/{r['name']}"
            rules.append(Rule(r["name"], ids, _compile(r.get("require_any", []), where), _compile(r["match_any"], where), _compile(r.get("veto_any", []), where)))
        contact = spec.get("contact_faq_id")
        if contact is not None and contact not in faq:
            raise FaqError(f"{intent}: contact_faq_id {contact!r} is not in the FAQ")
        result[intent] = IntentEntry(tuple(rules), contact, spec.get("note", ""))
    return result


class FaqResponder:
    def __init__(self, faq: Dict[str, FaqEntry], mapping: Dict[str, IntentEntry]):
        self.faq, self.mapping = faq, mapping

    def has_rules(self, intent: str) -> bool:
        return bool(self.mapping.get(intent) and self.mapping[intent].rules)

    def contact_note(self, intent: str) -> Optional[str]:
        """Approved contact line (verbatim FAQ text) for payroll-related intents, or None."""
        e = self.mapping.get(intent)
        if not e or not e.contact_faq_id:
            return None
        c = self.faq[e.contact_faq_id]
        return f"{c.question} {c.answer}"

    def answer(self, intent: str, cleaned_message: str) -> Optional[FaqAnswer]:
        """Deterministic: the matching rule with the most distinct match_any hits wins; ties go to the rule listed first."""
        entry = self.mapping.get(intent)
        if not entry:
            return None
        best: Optional[Tuple[int, int, Rule]] = None
        for order, rule in enumerate(entry.rules):
            if rule.require_any and not any(q.search(cleaned_message) for q in rule.require_any):
                continue
            if any(v.search(cleaned_message) for v in rule.veto_any):
                continue
            hits = sum(1 for m in rule.match_any if m.search(cleaned_message))
            if hits and (best is None or hits > best[0]):
                best = (hits, order, rule)
        if best is None:
            return None
        rule = best[2]
        answers = [self.faq[i] for i in rule.faq_ids]
        reply = answers[0].answer if len(answers) == 1 else "\n".join(f"{a.question} {a.answer}" for a in answers)
        return FaqAnswer(reply, rule.faq_ids, rule.name)

    def reachable_faq_ids(self) -> List[str]:
        return sorted({i for e in self.mapping.values() for r in e.rules for i in r.faq_ids})


def load_responder(faq_path: Path, mapping_path: Path, intents: Iterable[str]) -> FaqResponder:
    faq = load_faq(faq_path)
    return FaqResponder(faq, load_mapping(mapping_path, faq, intents))
