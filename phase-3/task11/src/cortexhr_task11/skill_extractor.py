"""Reusable dictionary-based skill extractor with a reviewed synonym table.

How it works: every skill has a canonical key (lower-case). Text is normalised (preprocessing.normalize_text) and each skill's
surface forms (the skill itself + synonyms) are matched with alphanumeric word boundaries, so 'sql' does not match inside
'postgresql' and 'java' does not match inside 'javascript'.

Vocabulary = BASE_SKILLS (generic technical skills named in the task brief) + every skill listed in the job descriptions
(+ optionally the structured `skills` column of the resumes, used ONLY to widen the vocabulary / validate the extractor).

Known, documented limitations (see README): matching is lexical, so a skill described only indirectly ('deployed a model
serving 80k predictions a day' for Model Deployment) is not found, and a longer phrase can credit a shorter skill
('Google Analytics' also matches the generic 'Analytics').
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .preprocessing import normalize_text

BASE_SKILLS: List[str] = [
    "Python", "Java", "SQL", "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch", "React", "Angular",
    "HTML", "CSS", "JavaScript", "AWS", "Docker", "Git", "Kubernetes", "Pandas", "scikit-learn", "Tableau", "Power BI", "Excel",
]

# canonical skill -> accepted surface variants (terminology / spelling / abbreviation). Equivalence only; no inference.
SYNONYMS: Dict[str, List[str]] = {
    "Excel": ["MS Excel", "Microsoft Excel"],
    "scikit-learn": ["sklearn", "scikit learn", "scikitlearn"],
    "Power BI": ["PowerBI", "Power-BI"],
    "CI/CD": ["CICD", "CI CD", "continuous integration"],
    "A/B Testing": ["AB Testing", "A B Testing", "split testing"],
    "REST APIs": ["REST API", "RESTful API", "RESTful APIs"],
    "NLP": ["natural language processing"],
    "PostgreSQL": ["Postgres"],
    "Go-to-Market": ["Go to Market", "GTM"],
    "Unit Testing": ["unit tests", "unit test"],
    "Kubernetes": ["K8s"],
    "AWS": ["Amazon Web Services"],
    "SEO": ["search engine optimization", "search engine optimisation"],
    "CRM": ["customer relationship management"],
    "Ticketing Tools": ["ticketing tool", "ticketing systems"],
    "Data Structures": ["data structure"],
    "Microservices": ["microservice", "micro services"],
    "Tally": ["Tally Prime"],
    "Data Visualization": ["data visualisation", "data viz"],
    "Email Campaigns": ["email campaign", "email marketing"],
    "MS Office": ["Microsoft Office"],
    "Problem Solving": ["problem-solving"],
    "Product Roadmaps": ["product roadmap"],
    "Lead Generation": ["lead gen"],
    "Google Analytics": ["GA4"],
    "JavaScript": ["java script"],
}


def skill_key(skill: str) -> str:
    """Case/punctuation-insensitive identity of a skill name (before synonym resolution)."""
    return normalize_text(skill)


class SkillExtractor:
    def __init__(self, vocabulary: Iterable[str] = (), synonyms: Optional[Dict[str, List[str]]] = None):
        self._synonyms: Dict[str, List[str]] = {k: list(v) for k, v in (SYNONYMS if synonyms is None else synonyms).items()}
        self._vocab: List[str] = []
        self._build(list(vocabulary))

    def _build(self, vocabulary: List[str]) -> None:
        # 1) synonym table first, so vocabulary entries such as 'MS Excel' resolve to the same key as 'Excel'
        self._alias_to_key: Dict[str, str] = {}
        for canon, alts in self._synonyms.items():
            ck = skill_key(canon)
            self._alias_to_key[ck] = ck
            for a in alts:
                self._alias_to_key[skill_key(a)] = ck
        # 2) vocabulary (first spelling seen becomes the display name)
        self._vocab = []
        self._display: Dict[str, str] = {}
        for s in list(vocabulary) + list(self._synonyms.keys()):
            k = self._alias_to_key.get(skill_key(s), skill_key(s))
            if k and k not in self._display:
                self._display[k] = str(s).strip() if skill_key(s) == k else self._display.get(k, str(s).strip())
            self._vocab.append(str(s).strip())
        # 3) one regex per key with all of its surface forms
        by_key: Dict[str, List[str]] = {}
        for alias, k in self._alias_to_key.items():
            by_key.setdefault(k, []).append(alias)
        self._patterns: Dict[str, re.Pattern] = {}
        for k in self._display:
            forms = sorted(set(by_key.get(k, []) + [k]), key=len, reverse=True)
            self._patterns[k] = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(f) for f in forms) + r")(?![a-z0-9])")

    @property
    def vocabulary(self) -> List[str]:
        return list(self._display.values())

    def canonical_key(self, skill: str) -> str:
        """Key used for matching; resolves synonyms ('MS Excel' and 'Excel' share one key). Unknown skills map to themselves."""
        k = skill_key(skill)
        return self._alias_to_key.get(k, k)

    def display_name(self, key: str) -> str:
        return self._display.get(key, key)

    def add_skills(self, skills: Iterable[str]) -> None:
        """Extend the vocabulary with skills not yet known (patterns are rebuilt)."""
        self._build(self._vocab + [s for s in skills if self.canonical_key(s) not in self._display])

    def extract_keys(self, text: str) -> List[str]:
        """Canonical keys of all vocabulary skills in `text`, ordered by first appearance."""
        s = normalize_text(text)
        hits: List[Tuple[int, str]] = []
        for k, pat in self._patterns.items():
            m = pat.search(s)
            if m:
                hits.append((m.start(), k))
        return [k for _, k in sorted(hits)]

    def evidence(self, text: str, key: str, width: int = 90) -> str:
        """The line of `text` that supports skill `key` (for explanations). '' if none."""
        pat = self._patterns.get(key)
        if pat is None:
            return ""
        for line in str(text).replace("\\n", "\n").splitlines():
            if pat.search(normalize_text(line)):
                return " ".join(line.split())[:width]
        return ""

    def extract(self, text: str) -> List[str]:
        """Display names of the skills found in `text`."""
        return [self._display[k] for k in self.extract_keys(text)]


def split_skill_list(value) -> List[str]:
    """Split a skills cell ('Python; SQL; Git') safely. Missing/blank -> []. Falls back to commas when no ';' is present."""
    if value is None or (isinstance(value, float) and value != value) or value is pd.NA:
        return []
    s = str(value).strip()
    if not s:
        return []
    parts = s.split(";") if ";" in s else (s.split(",") if "," in s else [s])
    seen, out = set(), []
    for p in parts:
        p = p.strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def build_default_extractor(jobs: Optional[pd.DataFrame] = None, resumes: Optional[pd.DataFrame] = None) -> SkillExtractor:
    """Vocabulary from the base list + job skill columns (+ resumes' structured `skills` column, vocabulary only)."""
    vocab: List[str] = list(BASE_SKILLS)
    if jobs is not None:
        for col in ("required_skills", "nice_to_have_skills"):
            if col in jobs.columns:
                for v in jobs[col]:
                    vocab.extend(split_skill_list(v))
    if resumes is not None and "skills" in resumes.columns:
        for v in resumes["skills"]:
            vocab.extend(split_skill_list(v))
    return SkillExtractor(vocab)
