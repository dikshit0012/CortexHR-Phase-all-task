"""Startup: validate settings, load the reference workbook once, load both artifacts once. Any problem is a StartupError."""
from __future__ import annotations

from .artifacts import ATTENDANCE, PAYROLL, artifact_path, load_artifact
from .config import Settings
from .reference import load_reference
from .scoring import AnomalyService


def load_service(settings: Settings) -> AnomalyService:
    reference = load_reference(settings.workbook_path)
    attendance = load_artifact(artifact_path(settings.model_dir, ATTENDANCE), ATTENDANCE, reference)
    payroll = load_artifact(artifact_path(settings.model_dir, PAYROLL), PAYROLL, reference)
    return AnomalyService(attendance, payroll, reference, settings.max_batch_size)
