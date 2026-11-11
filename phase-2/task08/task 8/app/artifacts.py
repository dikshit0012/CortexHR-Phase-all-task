"""Model artifacts: build (offline, once) and load (at startup, once).

What is saved - the ACTUAL fitted objects of the existing detectors, produced by their own fit() methods:
    attendance-anomaly-v1 : Task 5 `AttendanceAnomalyDetector`  -> SimpleImputer + IsolationForest + feature list
    payroll-anomaly-v1    : Task 6 `PayrollAnomalyModel`        -> SimpleImputer + IsolationForest + feature list
The Task 5 rules, the Task 6 rules and all Task 7 logic (reason codes, text, money at risk, ranking) are plain
Python code, not fitted state, so they are imported from Tasks 5-7 at run time and are NOT serialized.

Task 7 min-max scaling: Task 7 scales the Isolation Forest score over ALL scored rows of a dataset. The service
keeps that meaning by storing the min and max raw score of the reference data in the artifact (`score_scale`).
Every request is then scaled with the same two numbers, so a score never depends on which other records are in
the same request.

SECURITY: joblib uses pickle, so loading an artifact can run code. Only load artifacts built by this project.
"""
from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import openpyxl
import pandas as pd
import sklearn
import scipy

from . import _paths  # noqa: F401
from cortexhr_task5.attendance_anomaly_detector import (
    MODEL_FEATURES as ATTENDANCE_FEATURES, REQUIRED_COLUMNS as ATTENDANCE_REQUIRED,
    AttendanceAnomalyDetector, DetectorConfig, build_feature_frame,
)
from cortexhr_task6.payroll_features import (
    MODEL_FEATURES as PAYROLL_FEATURES, REQUIRED_COLUMNS as PAYROLL_REQUIRED, build_payroll_features,
)
from cortexhr_task6.payroll_model import ModelConfig, PayrollAnomalyModel

from .config import StartupError
from .reference import ReferenceData, fingerprint  # noqa: F401

FORMAT_VERSION = 2
ATTENDANCE = "attendance"
PAYROLL = "payroll"
VERSIONS = {ATTENDANCE: "attendance-anomaly-v1", PAYROLL: "payroll-anomaly-v1"}
_MODEL_FEATURES = {ATTENDANCE: ATTENDANCE_FEATURES, PAYROLL: PAYROLL_FEATURES}
_REQUIRED = {ATTENDANCE: ATTENDANCE_REQUIRED, PAYROLL: PAYROLL_REQUIRED}


@dataclass(frozen=True)
class ScoreScale:
    """Task 7 `normalise_scores` with the min / max fixed at build time instead of recomputed per call."""
    lo: float
    hi: float

    def apply(self, raw: pd.Series) -> pd.Series:
        s = pd.to_numeric(raw, errors="coerce")
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)) or self.hi == self.lo:
            return pd.Series(0.0, index=s.index)                   # same degenerate case as Task 7
        return ((s - self.lo) / (self.hi - self.lo)).clip(0.0, 1.0).round(3)   # clip: unseen rows can fall outside 0..1


@dataclass(frozen=True)
class LoadedDetector:
    kind: str
    version: str
    detector: Any                 # Task 5 AttendanceAnomalyDetector or Task 6 PayrollAnomalyModel (fitted)
    scale: ScoreScale
    metadata: Dict[str, Any]      # safe metadata only (no employee data, no file paths)


def artifact_path(model_dir: Path, kind: str) -> Path:
    return Path(model_dir) / f"{VERSIONS[kind]}.joblib"


def metadata_path(model_dir: Path, kind: str) -> Path:
    return Path(model_dir) / f"{VERSIONS[kind]}.metadata.json"


# --------------------------------------------------------------------------
# Build (offline)
# --------------------------------------------------------------------------
def _bundle(kind: str, det: Any, raw_scores: pd.Series, n_rows: int, ids_fp: str, created_at: str,
            workbook_name: str) -> Dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "detector": kind,
        "model_version": VERSIONS[kind],
        "created_at": created_at,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "joblib_version": joblib.__version__,
        "openpyxl_version": openpyxl.__version__,
        "config": asdict(det.config),
        "imputer": det.imputer_,
        "model": det.model_,
        "kept_features": list(det.kept_features_),
        "dropped_constant_features": list(det.dropped_constant_),
        "score_scale": {"min": float(raw_scores.min()), "max": float(raw_scores.max())},
        "reference": {"rows": int(n_rows), "id_fingerprint": ids_fp, "workbook_name": workbook_name},
        "input_schema": {"required_columns": list(_REQUIRED[kind]), "model_features": list(_MODEL_FEATURES[kind])},
    }


def build_artifacts(reference: ReferenceData, model_dir: Path, created_at: Optional[str] = None) -> Dict[str, Dict]:
    """Fit both detectors with their EXISTING training pipelines and save them. Returns the safe metadata."""
    created_at = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    fps = reference.fingerprints

    # Task 5 pipeline (run_detector): build features -> AttendanceAnomalyDetector(config).fit(features)
    att_feats = build_feature_frame(reference.attendance)
    att = AttendanceAnomalyDetector(DetectorConfig()).fit(att_feats)
    att_raw = att.predict(att_feats)["if_anomaly_score"]

    # Task 6 pipeline (run_detection): build features -> PayrollAnomalyModel(config).fit(features)
    pay_feats = build_payroll_features(reference.payroll)
    pay = PayrollAnomalyModel(ModelConfig()).fit(pay_feats)
    pay_raw = pay.predict(pay_feats)["if_anomaly_score"]

    bundles = {
        ATTENDANCE: _bundle(ATTENDANCE, att, att_raw, len(reference.attendance), fps["attendance"], created_at, reference.workbook_name),
        PAYROLL: _bundle(PAYROLL, pay, pay_raw, len(reference.payroll), fps["payroll"], created_at, reference.workbook_name),
    }
    out: Dict[str, Dict] = {}
    for kind, bundle in bundles.items():
        target = artifact_path(model_dir, kind)
        tmp = target.with_suffix(".joblib.tmp")
        joblib.dump(bundle, tmp, compress=3)
        os.replace(tmp, target)                                   # never leave a half-written artifact
        meta = {k: v for k, v in bundle.items() if k not in ("imputer", "model")}
        metadata_path(model_dir, kind).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        out[kind] = meta
    return out


# --------------------------------------------------------------------------
# Load (startup, once)
# --------------------------------------------------------------------------
_REQUIRED_KEYS = ("format_version", "detector", "model_version", "python_version", "numpy_version",
                  "sklearn_version", "pandas_version", "scipy_version", "joblib_version", "openpyxl_version",
                  "config", "imputer", "model", "kept_features", "dropped_constant_features", "score_scale",
                  "reference", "input_schema")


def load_artifact(path: Path, kind: str, reference: ReferenceData) -> LoadedDetector:
    """Load one artifact and verify it. Any problem raises StartupError - there is no dummy fallback."""
    path = Path(path)
    if kind not in VERSIONS:
        raise StartupError(f"unknown detector kind {kind!r}")
    hint = "Rebuild with: python scripts/build_artifacts.py"
    if not path.is_file():
        raise StartupError(f"Model artifact missing: {path.name}. {hint}")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise StartupError(f"Model artifact {path.name} could not be loaded ({type(exc).__name__}); it may be corrupted. {hint}") from None
    if not isinstance(bundle, dict):
        raise StartupError(f"Model artifact {path.name} has an unexpected structure. {hint}")
    missing = [k for k in _REQUIRED_KEYS if k not in bundle]
    if missing:
        raise StartupError(f"Model artifact {path.name} is incomplete (missing {missing}). {hint}")
    if bundle["format_version"] != FORMAT_VERSION:
        raise StartupError(f"Model artifact {path.name} has format {bundle['format_version']!r}, this service needs {FORMAT_VERSION}. {hint}")
    if bundle["detector"] != kind or bundle["model_version"] != VERSIONS[kind]:
        raise StartupError(f"Model artifact {path.name} is {bundle['detector']}/{bundle['model_version']}, "
                           f"expected {kind}/{VERSIONS[kind]}. Artifacts must not be mixed up.")
    compatible_versions = (
        ("Python", bundle["python_version"], platform.python_version()),
        ("NumPy", bundle["numpy_version"], np.__version__),
        ("pandas", bundle["pandas_version"], pd.__version__),
        ("SciPy", bundle["scipy_version"], scipy.__version__),
        ("scikit-learn", bundle["sklearn_version"], sklearn.__version__),
        ("joblib", bundle["joblib_version"], joblib.__version__),
        ("openpyxl", bundle["openpyxl_version"], openpyxl.__version__),
    )
    for library, built_with, installed in compatible_versions:
        if built_with != installed:
            raise StartupError(
                f"Model artifact {path.name} was built with {library} {built_with} but this environment has "
                f"{installed}. {hint}"
            )
    ref_fp = reference.fingerprints["attendance" if kind == ATTENDANCE else "payroll"]
    if bundle["reference"].get("id_fingerprint") != ref_fp:
        raise StartupError(f"Model artifact {path.name} was built from a different reference workbook than the configured one "
                           f"(its score scale would not match). {hint}")

    imputer, model, kept = bundle["imputer"], bundle["model"], bundle["kept_features"]
    if not (hasattr(imputer, "transform") and hasattr(model, "decision_function") and hasattr(model, "predict")):
        raise StartupError(f"Model artifact {path.name} does not contain a usable imputer and Isolation Forest. {hint}")
    if not kept or not set(kept) <= set(_MODEL_FEATURES[kind]):
        raise StartupError(f"Model artifact {path.name} lists features that are not in the {kind} feature contract. {hint}")
    if getattr(imputer, "n_features_in_", len(kept)) != len(kept) or getattr(model, "n_features_in_", len(kept)) != len(kept):
        raise StartupError(f"Model artifact {path.name} has preprocessing / model inputs that do not match its feature list. {hint}")
    lo, hi = bundle["score_scale"]["min"], bundle["score_scale"]["max"]
    if not (isinstance(lo, float) and isinstance(hi, float) and np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise StartupError(f"Model artifact {path.name} has an invalid score scale. {hint}")

    try:
        if kind == ATTENDANCE:
            det = AttendanceAnomalyDetector(DetectorConfig(**bundle["config"]))
        else:
            det = PayrollAnomalyModel(ModelConfig(**bundle["config"]))
    except TypeError:
        raise StartupError(f"Model artifact {path.name} has a configuration this code version does not understand. {hint}") from None
    det.imputer_, det.model_ = imputer, model
    det.kept_features_, det.dropped_constant_ = list(kept), list(bundle["dropped_constant_features"])
    metadata = {k: v for k, v in bundle.items() if k not in ("imputer", "model")}
    return LoadedDetector(kind=kind, version=bundle["model_version"], detector=det, scale=ScoreScale(lo, hi), metadata=metadata)
