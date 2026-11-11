"""Text-only personal-information masking for the fair screener.

Why: Task 11 removes name / e-mail / city / tier by looking up the VALUES of structured columns. If a resume arrives as text only
(or the columns are wrong) that protection disappears. This module masks personal information from the resume TEXT itself, with
general rules (no pair ids, no lists of known names), and the fair screener is only ever given `resume_id` + `resume_text`.

What is masked (each category can be switched off for the ablation in bias_detection.py):
  name         candidate name in the header line (and any other occurrence)           email     e-mail addresses and URLs
  phone        phone numbers                                                          location  the 'City |' prefix of the contact line
  tier         institution-tier statements ('Tier 2 institution'); the DEGREE text next to it is kept (e.g. 'M.Sc Statistics' is job-relevant)
  age          date of birth, 'age: 34', '34 years old'                               personal  gender / marital status / religion / caste / nationality fields
  title        gendered honorifics (Mr, Mrs, Miss, Shri, Smt, 'Ms.')                  pronoun   he / she / him / her / his / hers / himself / herself
  graduation   years (1900-2099), 'Month YYYY', 'class of' / 'batch of'

NOT masked on purpose: skills, certifications, degree subjects, job titles, employers, durations ('2.2 yrs'), and the explicit
'N years of experience' statement (experience is reported, not scored). 'MS Excel' / 'MS Office' are kept: only the dotted 'Ms.' is a title.
"""
from __future__ import annotations

import re
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

ALL_CATEGORIES: FrozenSet[str] = frozenset(
    {"name", "email", "phone", "location", "tier", "age", "personal", "title", "pronoun", "graduation"})

# Columns that may exist next to a resume but must NEVER reach scoring.
AUDIT_ONLY_COLUMNS = frozenset({"inferred_gender", "gender", "age", "date_of_birth", "dob", "college_tier", "candidate_name", "email",
                                "city", "highest_education", "photo", "religion", "caste"})
SCORING_INPUT_COLUMNS = ("resume_id", "resume_text")

_SECTIONS = {"summary", "experience", "skills", "education", "certifications", "profile", "objective", "experience notes", "resume",
             "curriculum vitae", "contact", "projects"}
_ROLE_WORDS = {"engineer", "developer", "manager", "analyst", "executive", "accountant", "scientist", "designer", "consultant", "associate",
               "intern", "trainee", "coordinator", "representative", "specialist", "officer", "administrator", "writer", "resume", "vitae", "cv"}
_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z.'\-]+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.(?:example|com|in|org|net|co|edu)", re.I)
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{8,}\d(?!\w)")
_LOCATION = re.compile(r"(?m)^[ \t]*[A-Za-z][A-Za-z .]{1,30}?[ \t]*\|[ \t]*(?=[\w.+-]+@)")
_TIER_INLINE = re.compile(r"(?i)\btier[\s-]*(?:[0-9]+|i{1,3}|iv)\b(?:\s*(?:institution|institute|college|university))?")
_DOB = re.compile(r"(?i)\b(?:date\s+of\s+birth|d\.?o\.?b\.?|born(?:\s+(?:on|in))?)\b[^\n]*")
_AGE = re.compile(r"(?i)\bage\s*[:\-]?\s*\d{1,2}(?:\s*(?:years?|yrs?)(?:[\s-]*old)?)?\b|\b\d{1,2}\s*(?:years?|yrs?)[\s-]*old\b")
_PERSONAL = re.compile(r"(?i)\b(?:gender|sex|marital\s+status|religion|caste|nationality)\b\s*[:\-][^\n]*")
_PERSONAL_HEADING = re.compile(r"(?im)^[ \t]*personal(?:\s+(?:details|information|info|profile|data))?[ \t]*:?[ \t]*$")
_TITLE = re.compile(r"(?i)\b(?:mr|mrs|miss|shri|smt|shrimati|kumari)\b\.?|\bms\.")
_PRONOUN = re.compile(r"(?i)\b(?:he|she|him|her|hers|his|himself|herself)\b")
_MONTH_YEAR = re.compile(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*,?\s*(?:19|20)\d{2}\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_CLASS_OF = re.compile(r"(?i)\b(?:class|batch)\s+of\b")


def _header_name(text: str) -> Optional[str]:
    """Candidate name from the first non-blank line, only when it looks like a name (accepts two-column layouts: text before 2+ spaces)."""
    for line in text.splitlines():
        if not line.strip():
            continue
        chunk = re.split(r"\s{2,}|\t", line.strip())[0].strip()
        toks = chunk.split()
        if not (2 <= len(toks) <= 4) or not all(_NAME_TOKEN.fullmatch(t) for t in toks):
            return None
        low = {t.lower().strip(".") for t in toks}
        if chunk.lower() in _SECTIONS or low & _SECTIONS or low & _ROLE_WORDS:
            return None
        emails = [e.lower().split("@")[0] for e in _EMAIL.findall(text)]
        confirmed = any(t.lower().strip(".") in local for local in emails for t in toks)
        looks_like_name = chunk.isupper() or all(t[0].isupper() for t in toks)
        return chunk if (confirmed or looks_like_name) else None
    return None


def mask_personal_information(text, categories: Iterable[str] = ALL_CATEGORIES, extra_values: Iterable = ()) -> Tuple[str, Dict[str, int]]:
    """Return (masked_text, {category: number_of_masked_items}). Never raises on missing / odd input."""
    cats = set(categories)
    unknown = cats - ALL_CATEGORIES
    if unknown:
        raise ValueError(f"unknown categories: {sorted(unknown)}")
    if text is None or (isinstance(text, float) and text != text):
        return "", {}
    s = str(text).replace("\\n", "\n")
    counts: Dict[str, int] = {}

    def sub(cat: str, pat: re.Pattern, repl: str = " ") -> None:
        nonlocal s
        if cat in cats:
            s, n = pat.subn(repl, s)
            if n:
                counts[cat] = counts.get(cat, 0) + n

    if "name" in cats:
        names = [n for n in ([_header_name(s)] + [str(v) for v in extra_values if v == v and v is not None]) if n and len(n) >= 2]
        for nm in names:
            pat = re.compile(r"(?<![A-Za-z0-9])" + r"\s+".join(re.escape(p) for p in nm.split()) + r"(?![A-Za-z0-9])", re.I)
            s, n = pat.subn(" ", s)
            if n:
                counts["name"] = counts.get("name", 0) + n
    sub("location", _LOCATION)                     # must run before e-mail removal (it anchors on the e-mail)
    sub("email", _EMAIL)
    sub("email", _URL)
    sub("phone", _PHONE)
    sub("tier", _TIER_INLINE)
    sub("age", _DOB)
    sub("age", _AGE)
    sub("personal", _PERSONAL)
    sub("personal", _PERSONAL_HEADING)
    sub("title", _TITLE)
    sub("pronoun", _PRONOUN)
    sub("graduation", _MONTH_YEAR)
    sub("graduation", _YEAR)
    sub("graduation", _CLASS_OF)
    return s, counts
