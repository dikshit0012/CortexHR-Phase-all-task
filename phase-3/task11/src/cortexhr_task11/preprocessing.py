"""Resume / job text preprocessing and (safe) experience extraction.

Design notes
* Same normalisation for every text (resumes and job descriptions), deterministic, never raises on odd input.
* Task 9's `clean_text` was reviewed. Its conventions (NFKC, lowercase, whitespace collapse, NaN-safe) are followed here, but it
  is NOT imported: it applies a chatbot typo-correction dictionary built from FAQ data. Importing it would also couple Task 11 to Task 9's folder layout.
* Identity / proxy scrubbing: supplied name/contact/demographic/college values and common free-text variants are removed before
  TF-IDF and skill extraction. Text-only redaction is pattern-based; it is not a universal PII detector.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Iterable, Optional, Tuple

_KEEP = re.compile(r"[^a-z0-9+#/&.\-]+")
_SPACES = re.compile(r"\s+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.(?:example|com|in|org|net|co)", re.I)
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_NAME_LABEL = re.compile(r"\b(?:candidate\s+)?name\s*[:=-]\s*[^|,;\r\n]+", re.I)
_TIER = re.compile(r"\btier\s*[-:]?\s*(?:[1-9]|i{1,3})(?:\s*(?:college|university|institution|school))?\b", re.I)
_PRESTIGE = re.compile(
    r"\b(?:(?:top[- ]tier|prestigious|elite|premier|highly ranked|ivy league)"
    r"(?:\s+(?:university|college|institution|school))?)\b",
    re.I,
)
_INSTITUTION_NAME = re.compile(
    r"\b(?:(?:[A-Z][A-Za-z0-9&'.-]*|[A-Z]{2,})\s+){1,5}"
    r"(?i:university|college|institute|school)\b"
)
_UNIVERSITY_OF = re.compile(
    r"\b(?i:university|college|institute|school)\s+(?i:of)\s+"
    r"(?:[A-Z][A-Za-z&'.-]*|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z&'.-]*|[A-Z]{2,})){0,3}"
    r"(?:,\s*(?:[A-Z][A-Za-z&'.-]*|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z&'.-]*|[A-Z]{2,})){0,2})?"
)
_KNOWN_INSTITUTIONS = re.compile(
    r"\b(?:(?:IIT|NIT|IIM)\s+[A-Za-z]+(?:\s+[A-Za-z]+)?|"
    r"BITS(?:\s+Pilani)?|Stanford|Harvard|MIT|Oxford|Cambridge|Yale|Princeton|Columbia|Berkeley|"
    r"Caltech|Cornell|Duke|Wharton|IIT|NIT|IIM|UCL|LSE)\b",
    re.I,
)
_AGE = re.compile(
    r"\b(?:age|aged)\s*[:=]?\s*\d{1,3}\b"
    r"|\b\d{1,3}\s*[- ]\s*years?\s*[- ]\s*old\b"
    r"|\b\d{1,3}\s+years?\s+old\b"
    r"|\b\d{1,3}\s*y/o\b"
    r"|\b(?:date[- ]of[- ]birth|birth date|birthdate|dob|born(?:\s+in)?)\s*[:=]?\s*"
    r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:19|20)\d{2})\b",
    re.I,
)
_DEMOGRAPHIC_LABEL = re.compile(r"\b(?:gender|sex|pronouns?)\s*[:=]\s*[^|,;.\r\n]+", re.I)
_GENDER = re.compile(
    r"\b(?:male|female|man|woman|men|women|non[- ]?binary|transgender|cisgender|"
    r"gender(?:ed)?|pronouns?|he|him|his|she|her|hers|they|them|their|theirs)\b",
    re.I,
)
_NUMERIC_TOKEN = re.compile(r"^[0-9.]+k?$")
_EXP_STATEMENT = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+of\s+(?:\w+\s+){0,2}?experience")
_ROLE_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*yrs?\b")
MAX_PLAUSIBLE_YEARS = 50.0

EXP_SUMMARY = "summary_statement"
EXP_ROLES = "role_durations_sum"
EXP_NONE = "not_found"


def _as_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    try:  # pandas.NA / NaT
        import pandas as pd
        if text is pd.NA or text is pd.NaT:
            return ""
    except Exception:  # pragma: no cover
        pass
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    return text if isinstance(text, str) else str(text)


def normalize_text(text: Any) -> str:
    """Lowercase, NFKC, drop special characters except those that are part of skill names (+ # / & . -), collapse whitespace."""
    s = unicodedata.normalize("NFKC", _as_text(text)).lower()
    s = s.replace("\\n", " ")
    s = _KEEP.sub(" ", s)
    toks = [t.strip(".-/&") for t in s.split()]
    return _SPACES.sub(" ", " ".join(t for t in toks if t)).strip()


def scrub_identity(text: Any, identity_values: Iterable[Any] = ()) -> str:
    """Remove demographic, contact and supplied identity/proxy information from resume text."""
    s = _as_text(text)
    s = _EMAIL.sub(" ", s)
    s = _URL.sub(" ", s)
    s = _NAME_LABEL.sub(" ", s)
    s = _DEMOGRAPHIC_LABEL.sub(" ", s)
    s = _TIER.sub(" ", s)
    s = _PRESTIGE.sub(" ", s)
    s = _UNIVERSITY_OF.sub(" ", s)
    s = _INSTITUTION_NAME.sub(" ", s)
    s = _KNOWN_INSTITUTIONS.sub(" ", s)
    s = _AGE.sub(" ", s)
    s = _GENDER.sub(" ", s)
    for v in identity_values:
        v = _as_text(v).strip()
        if len(v) >= 2 and not re.fullmatch(r"[\d./ -]+", v):
            pat = r"(?<![A-Za-z0-9])" + r"\s+".join(re.escape(p) for p in v.split()) + r"(?![A-Za-z0-9])"
            s = re.sub(pat, " ", s, flags=re.I)
    return s


def tfidf_text(text: Any) -> str:
    """Normalised text without numeric tokens ('28', '87k', '4.8') and duration markers, which carry no skill meaning."""
    toks = [t for t in normalize_text(text).split() if not _NUMERIC_TOKEN.match(t) and t not in {"yrs", "yr"}]
    return " ".join(toks)


def extract_experience_years(text: Any) -> Tuple[Optional[float], str]:
    """Years of experience, only when the resume states it.

    1. An explicit statement such as 'with 4.8 years of experience' -> that number.
    2. Otherwise the sum of explicit role durations written like '(2.2 yrs)'.
    3. Otherwise (None, 'not_found'). A bare number elsewhere in the text is never interpreted as experience.
    """
    s = normalize_text(text)
    m = _EXP_STATEMENT.search(s)
    if m:
        v = float(m.group(1))
        if 0 <= v <= MAX_PLAUSIBLE_YEARS:
            return v, EXP_SUMMARY
    durs = [float(x) for x in _ROLE_DURATION.findall(s)]
    if durs:
        total = round(sum(durs), 2)
        if 0 <= total <= MAX_PLAUSIBLE_YEARS:
            return total, EXP_ROLES
    return None, EXP_NONE
