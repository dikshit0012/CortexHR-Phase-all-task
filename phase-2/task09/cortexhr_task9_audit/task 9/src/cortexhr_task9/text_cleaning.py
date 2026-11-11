"""Reusable, deterministic text cleaning. The SAME function runs at training time and at inference time (it is the first
step of the saved scikit-learn pipeline), so the two can never drift apart.

What it does:   Unicode NFKC, lowercase, curly quotes -> straight, repeated whitespace -> one space, trim, collapse runs of
                the same punctuation mark ('???' -> '?'), and a SMALL reviewed typo dictionary (typo_corrections.json,
                built from the training data: one adjacent-letter swap of a more frequent word, e.g. 'laeve' -> 'leave').
What it keeps:  negations, digits, dates / month names, names, '&', question marks - nothing is stripped as a stop word, so
                'not', 'no', 'form 16' or 'sept' survive. Filler such as 'pls' / 'urgently' is kept as well: the model
                learns its weight itself and removing it could erase real signal.
Never raises on odd input: None / NaN / numbers / bytes give '' or their string form.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

CLEANING_VERSION = "clean-v1"          # stored in the model artifact; a mismatch makes the loader refuse the artifact
TYPO_FILE = Path(__file__).with_name("typo_corrections.json")

_SPACES = re.compile(r"\s+")
_REPEAT_PUNCT = re.compile(r"([?!.,;:])\1+")
_TOKEN = re.compile(r"[a-z0-9&']+")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00b4": "'", "`": "'"})


@lru_cache(maxsize=1)
def load_typo_map() -> Dict[str, str]:
    return json.loads(TYPO_FILE.read_text(encoding="utf-8"))


def clean_text(text: Any) -> str:
    """Return the cleaned form of one message ('' for missing / blank input)."""
    if text is None:
        return ""
    if isinstance(text, float) and text != text:            # NaN
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    elif not isinstance(text, str):
        text = str(text)
    s = unicodedata.normalize("NFKC", text).lower().translate(_QUOTES)
    s = _REPEAT_PUNCT.sub(r"\1", s)
    typos = load_typo_map()
    s = _TOKEN.sub(lambda m: typos.get(m.group(0), m.group(0)), s)
    return _SPACES.sub(" ", s).strip()


def clean_texts(texts: Iterable[Any]) -> List[str]:
    """Vectorised wrapper used as the first step of the sklearn Pipeline (must be a module-level function to pickle)."""
    return [clean_text(t) for t in texts]
