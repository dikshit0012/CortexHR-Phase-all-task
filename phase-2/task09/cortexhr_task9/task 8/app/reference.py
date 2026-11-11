"""Reference workbook: the history the Task 5 / Task 6 features need (earlier months, role medians, duplicates).

Only the two feature sheets are read. Label and answer-key sheets are never opened.
The reference is loaded once at startup and then only read.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from . import _paths  # noqa: F401  (puts Task 5 / 6 / 7 on sys.path)
from cortexhr_task5.attendance_anomaly_detector import ID_COL as ATTENDANCE_ID, validate_attendance_frame
from cortexhr_task6.payroll_features import ID_COL as PAYROLL_ID, validate_payroll_frame

from .config import ENV_WORKBOOK, StartupError

ATTENDANCE_SHEET = "attendance_monthly_features"
PAYROLL_SHEET = "payroll_monthly_features"


@dataclass(frozen=True)
class ReferenceData:
    attendance: pd.DataFrame      # validated raw rows (all original columns), never modified after loading
    payroll: pd.DataFrame
    workbook_name: str            # file name only (no directory)

    @property
    def fingerprints(self) -> dict:
        return {"attendance": fingerprint(self.attendance[ATTENDANCE_ID]),
                "payroll": fingerprint(self.payroll[PAYROLL_ID])}


def fingerprint(ids: Iterable) -> str:
    """Hash of the sorted record ids: detects an artifact that was built from a different reference."""
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode()).hexdigest()


def validate_workbook_path(path: Optional[Path]) -> Path:
    if path is None:
        raise StartupError(f"{ENV_WORKBOOK} is not set. Point it at the reference workbook (02_CortexHR_AI_ML_Dataset.xlsx).")
    p = Path(path).expanduser()
    if not p.exists():
        raise StartupError(f"Reference workbook not found: {p} (check {ENV_WORKBOOK}).")
    if not p.is_file():
        raise StartupError(f"{ENV_WORKBOOK} must point to a file, not a directory: {p}")
    if p.suffix.lower() != ".xlsx":
        raise StartupError(f"Reference workbook must be an .xlsx file, got {p.suffix or 'no extension'!r}: {p.name}")
    if p.stat().st_size == 0:
        raise StartupError(f"Reference workbook is empty: {p.name}")
    return p.resolve()


def load_reference(path: Optional[Path]) -> ReferenceData:
    p = validate_workbook_path(path)
    try:
        book = pd.ExcelFile(p)
    except Exception as exc:                                   # corrupt file, not a real xlsx, ...
        raise StartupError(f"Reference workbook {p.name} could not be read ({type(exc).__name__}).") from None
    with book:
        missing = [name for name in (ATTENDANCE_SHEET, PAYROLL_SHEET) if name not in book.sheet_names]
        if missing:
            raise StartupError(f"Reference workbook {p.name} is missing a required sheet: {missing}.")
        try:
            sheets = {name: book.parse(name) for name in (ATTENDANCE_SHEET, PAYROLL_SHEET)}   # only these two sheets are read
        except Exception as exc:
            raise StartupError(f"Reference workbook {p.name} could not be read ({type(exc).__name__}).") from None
    try:
        attendance = validate_attendance_frame(sheets[ATTENDANCE_SHEET])
        payroll = validate_payroll_frame(sheets[PAYROLL_SHEET])
    except ValueError as exc:
        raise StartupError(f"Reference workbook {p.name} failed validation: {exc}") from None
    return ReferenceData(attendance=attendance, payroll=payroll, workbook_name=p.name)
