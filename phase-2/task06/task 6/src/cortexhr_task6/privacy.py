"""Task 6 - masking of identifiers in anything that is saved for review."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets

import pandas as pd

_PROCESS_SALT = secrets.token_bytes(32)


def _salt() -> bytes:
    configured = os.environ.get("CORTEXHR_ID_SALT")
    return configured.encode() if configured else _PROCESS_SALT


def mask_ids(values: pd.Series, prefix: str) -> pd.Series:
    """Keyed hash of each id -> stable reference such as 'EMP-3fa9c1d2a71b4e80'.

    Stable within one process and across its output files. The default key is
    random per process; set CORTEXHR_ID_SALT to a strong secret for stable
    references across runs. Keep configured secrets out of the repository.
    """
    def one(v) -> str:
        digest = hmac.new(_salt(), str(v).encode(), hashlib.sha256).hexdigest()[:16]
        return f"{prefix}-{digest}"
    return values.map(one)
