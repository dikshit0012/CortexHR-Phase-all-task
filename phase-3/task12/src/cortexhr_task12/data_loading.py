"""Twin pairs (public design table) and protected attributes (confidential unlock_CX-12; used ONLY to audit, never to score)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import paths

PAIR_COLUMNS = ["pair_id", "resume_id_original", "resume_id_twin", "attribute_changed", "skills_identical", "experience_identical",
                "expected_score_difference"]
PROTECTED_COLUMNS = ["resume_id", "inferred_gender", "college_tier"]


class AuditDataError(ValueError):
    pass


def load_pairs(path: Optional[Path] = None, resumes: Optional[pd.DataFrame] = None, expected_count: Optional[int] = None) -> pd.DataFrame:
    p = pd.read_csv(path or paths.PAIRS_CSV)
    miss = [c for c in PAIR_COLUMNS if c not in p.columns]
    if miss:
        raise AuditDataError(f"resume_bias_pairs: missing columns {miss}")
    if p.empty:
        raise AuditDataError("resume_bias_pairs contains no pairs")
    for column in PAIR_COLUMNS:
        if p[column].isna().any() or p[column].astype(str).str.strip().eq("").any():
            raise AuditDataError(f"resume_bias_pairs: missing values in {column}")
    if expected_count is not None and len(p) != expected_count:
        raise AuditDataError(f"resume_bias_pairs: expected {expected_count} pairs, found {len(p)}")
    if p["pair_id"].duplicated().any():
        raise AuditDataError("duplicate pair_id")
    ids = pd.concat([p["resume_id_original"], p["resume_id_twin"]])
    if ids.duplicated().any():
        raise AuditDataError("a resume appears in more than one pair / twice in a pair")
    if (p["resume_id_original"] == p["resume_id_twin"]).any():
        raise AuditDataError("a pair links a resume to itself")
    try:
        expected = pd.to_numeric(p["expected_score_difference"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise AuditDataError("expected_score_difference must be numeric") from exc
    if not expected.map(lambda value: pd.notna(value) and value >= 0 and value < float("inf")).all():
        raise AuditDataError("expected_score_difference must contain finite non-negative values")
    if resumes is not None:
        if "resume_id" not in resumes.columns or resumes["resume_id"].isna().any() or resumes["resume_id"].duplicated().any():
            raise AuditDataError("resumes must contain unique, non-missing resume_id values")
        unknown = sorted(set(ids) - set(resumes["resume_id"]))
        if unknown:
            raise AuditDataError(f"pair resumes not in resumes table: {unknown[:5]}")
    return p.reset_index(drop=True)


def load_protected(path: Optional[Path] = None, required: bool = False) -> Optional[pd.DataFrame]:
    """Returns None when the confidential file is absent (the score audit still runs; protected-value analyses are skipped)."""
    f = Path(path) if path else paths.PROTECTED_CSV
    if not f.exists():
        if required:
            raise FileNotFoundError(f"{f} not found (confidential unlock_CX-12; set CORTEXHR_TASK12_PRIVATE)")
        return None
    d = pd.read_csv(f)
    miss = [c for c in PROTECTED_COLUMNS if c not in d.columns]
    if miss:
        raise AuditDataError(f"protected attributes: missing columns {miss}")
    if d["resume_id"].duplicated().any():
        raise AuditDataError("protected attributes: duplicate resume_id")
    if d["resume_id"].isna().any():
        raise AuditDataError("protected attributes: missing resume_id")
    return d
