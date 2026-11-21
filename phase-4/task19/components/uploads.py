"""Validation for user-supplied files and tables. Nothing unsupported is ever sent to the API."""
from __future__ import annotations

import io
import math

import pandas as pd

MAX_CSV_BYTES = 700_000
MAX_ROWS = 100
MAX_TXT_BYTES = 40_000        # API limit is 20,000 characters per resume
MAX_RESUME_CHARS = 20_000
MAX_RESUMES = 50
IDENTITY_COLUMNS = {"name", "full_name", "first_name", "last_name", "email", "phone", "address", "dob", "date_of_birth",
                    "gender", "age", "national_id", "ssn", "pan", "aadhaar"}


class UploadError(Exception):
    """Message is safe to show to the user."""


def _ext(name: str) -> str:
    return name.lower().rsplit(".", 1)[-1] if "." in name else ""


def read_csv_upload(name: str, data: bytes) -> pd.DataFrame:
    if _ext(name) != "csv":
        raise UploadError("Please upload a .csv file.")
    if len(data) > MAX_CSV_BYTES:
        raise UploadError(f"That file is too large (limit {MAX_CSV_BYTES // 1000} KB).")
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise UploadError("The file has no rows.")
    except Exception:
        raise UploadError("That file could not be read as a CSV table.")
    if df.empty:
        raise UploadError("The file has no rows.")
    if len(df) > MAX_ROWS:
        raise UploadError(f"The file has {len(df)} rows; the limit is {MAX_ROWS} per request.")
    return df


def table_to_records(df: pd.DataFrame, fields: dict, extra_ok: set[str] = frozenset(), required: set[str] = frozenset()) -> list[dict]:
    """Check columns against the API schema, convert types, return clean records. Raises UploadError on any problem."""
    cols = [c.strip() for c in df.columns]
    df = df.copy(); df.columns = cols
    bad_identity = sorted(c for c in cols if c.lower() in IDENTITY_COLUMNS)
    if bad_identity:
        raise UploadError(f"Remove column(s) {', '.join(bad_identity)}: names and personal identifiers are not accepted.")
    unknown = sorted(set(cols) - set(fields) - set(extra_ok))
    if unknown:
        raise UploadError(f"Unknown column(s): {', '.join(unknown)}. Use only the documented fields.")
    missing = sorted((required | {k for k, v in fields.items() if v.get("required")}) - set(cols))
    if missing:
        raise UploadError(f"Missing required column(s): {', '.join(missing)}.")
    records = []
    for i, row in enumerate(df.to_dict("records"), start=1):
        rec = {}
        for k, raw in row.items():
            raw = (raw or "").strip()
            meta = fields.get(k, {"type": "string"})
            if raw == "":
                rec[k] = None
                continue
            t = meta.get("type")
            try:
                if t in ("integer",):
                    f = float(raw)
                    if f != int(f):
                        raise ValueError
                    rec[k] = int(f)
                elif t == "number":
                    f = float(raw)
                    if math.isnan(f) or math.isinf(f):
                        raise ValueError
                    rec[k] = f
                else:
                    rec[k] = raw
            except ValueError:
                raise UploadError(f"Row {i}, column {k}: expected a {'whole number' if t == 'integer' else 'number'}.")
            if t in ("integer", "number"):
                lo, hi = meta.get("min"), meta.get("max")
                if (lo is not None and rec[k] < lo) or (hi is not None and rec[k] > hi):
                    raise UploadError(f"Row {i}, column {k}: value must be between {lo:g} and {hi:g}.")
        records.append({k: v for k, v in rec.items() if v is not None or fields.get(k, {}).get("required")})
    return records


def read_resume_txt(name: str, data: bytes) -> str:
    if _ext(name) != "txt":
        raise UploadError(f"{name}: only .txt files are accepted (PDF and Word files are not supported by the API).")
    if len(data) > MAX_TXT_BYTES:
        raise UploadError(f"{name}: file is too large.")
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise UploadError(f"{name}: the file is not UTF-8 text.")
    if not text:
        raise UploadError(f"{name}: the file is empty.")
    if len(text) > MAX_RESUME_CHARS:
        raise UploadError(f"{name}: text is longer than {MAX_RESUME_CHARS:,} characters.")
    return text
