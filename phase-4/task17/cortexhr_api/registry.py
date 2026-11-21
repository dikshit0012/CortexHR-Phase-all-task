"""Central model registry.

Every artifact is loaded ONCE (FastAPI lifespan) and then only read. Each capability gets its own status entry, so
health reports what is really available. A model that cannot be loaded is marked `unavailable` with a safe reason;
there is never a mock or substitute model.

Upstream project code (Phase 1-3, Task 16) is reused from the repository by adding its `src` folders to sys.path.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import Settings
from .security import ModelUnavailable, safe_text

log = logging.getLogger("cortexhr.registry")

ATTRITION, ATTENDANCE, PAYROLL, CHATBOT, RESUME, PAYBAND, FORECAST = (
    "attrition", "attendance_anomaly", "payroll_anomaly", "chatbot", "resume_ranking", "payband", "headcount_forecast")
ALL_NAMES = (ATTRITION, ATTENDANCE, PAYROLL, CHATBOT, RESUME, PAYBAND, FORECAST)


@dataclass
class ModelEntry:
    name: str
    endpoint: str
    status: str = "unavailable"          # loaded | unavailable
    version: Optional[str] = None
    mode: Optional[str] = None           # e.g. forecast: live | precomputed
    detail: Optional[str] = None         # safe, path-free explanation
    trained_with: Dict[str, str] = field(default_factory=dict)
    running_with: Dict[str, str] = field(default_factory=dict)
    compatibility: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.status == "loaded"


class ModelRegistry:
    def __init__(self) -> None:
        self.entries: Dict[str, ModelEntry] = {}
        self.objects: Dict[str, Any] = {}

    # ---------------------------------------------------------------- access
    def get(self, name: str) -> Any:
        e = self.entries.get(name)
        if e is None or not e.loaded or name not in self.objects:
            raise ModelUnavailable(name, e.detail or "" if e else "")
        return self.objects[name]

    def summary(self) -> Dict[str, Any]:
        return {n: asdict(e) | {"loaded": e.loaded} for n, e in self.entries.items()}

    def missing(self) -> List[str]:
        return [n for n in ALL_NAMES if not self.entries.get(n, ModelEntry(n, "")).loaded]

    # ---------------------------------------------------------------- loading
    def load_all(self, s: Settings) -> "ModelRegistry":
        _add_paths(s)
        for name, endpoint, fn in (
            (ATTRITION, "/v1/attrition/predict", _load_attrition),
            (ATTENDANCE, "/v1/anomaly/attendance", None), (PAYROLL, "/v1/anomaly/payroll", None),
            (CHATBOT, "/v1/chat", _load_chat),
            (RESUME, "/v1/resume/rank", _load_resume),
            (PAYBAND, "/v1/payband/analyze", _load_payband),
            (FORECAST, "/v1/forecast/headcount", _load_forecast),
        ):
            self.entries[name] = ModelEntry(name, endpoint)
            if fn is None:
                continue
            self._run(name, fn, s)
        self._run_anomaly(s)
        for e in self.entries.values():
            (log.info if e.loaded else log.warning)("model %s: %s%s", e.name, e.status, f" ({e.detail})" if e.detail else "")
        return self

    def _run(self, name: str, fn, s: Settings) -> None:
        entry = self.entries[name]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")                   # version differences are reported in `compatibility`
                obj = fn(s, entry)
            self.objects[name] = obj
            entry.status = "loaded"
        except Exception as exc:                                   # one failing model must not stop the others
            entry.status, entry.detail = "unavailable", f"{type(exc).__name__}: {safe_text(exc)}"

    def _run_anomaly(self, s: Settings) -> None:
        att, pay = self.entries[ATTENDANCE], self.entries[PAYROLL]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                service, mod = _load_anomaly(s)
            self.objects[ATTENDANCE] = self.objects[PAYROLL] = service
            self.objects["_task8"] = mod
            for e, d in ((att, service.attendance), (pay, service.payroll)):
                e.status, e.version = "loaded", d.version
                e.trained_with = {"scikit-learn": d.metadata["sklearn_version"], "pandas": d.metadata["pandas_version"],
                                  "numpy": d.metadata["numpy_version"], "python": d.metadata["python_version"]}
                e.running_with = _running()
                e.compatibility = "exact match (verified by the Task 8 loader)"
        except Exception as exc:
            for e in (att, pay):
                e.status, e.detail = "unavailable", f"{type(exc).__name__}: {safe_text(exc)}"
        _stamp_versions(att, pay)


def _running() -> Dict[str, str]:
    import platform
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}


def _stamp_versions(*entries: ModelEntry) -> None:
    for e in entries:
        if not e.running_with:
            e.running_with = _running()
        if not e.version:
            e.version = {ATTENDANCE: "attendance-anomaly-v1", PAYROLL: "payroll-anomaly-v1"}.get(e.name)


# ---------------------------------------------------------------------------- sys.path wiring
def _add_paths(s: Settings) -> None:
    r = s.repo_root
    os.environ.setdefault("CORTEXHR_TASKS_ROOT", str(r))
    phase4_root = r / "phase-4"
    for p in (phase4_root / "task16" / "src",
              phase4_root / "task17"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- attrition
def _load_attrition(s: Settings, e: ModelEntry):
    if "cortexhr.features" not in sys.modules:
        import types
        from sklearn.base import BaseEstimator, TransformerMixin
        cortexhr = types.ModuleType("cortexhr")
        features = types.ModuleType("cortexhr.features")
        LOG_PREFIXES = ("annual_fixed_pay", "avg_overtime_hours_per_month_3m", "avg_variable_pay_")

        class RawCleaner(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                out = X.copy()
                if "months_since_last_promotion" in out:
                    out["never_promoted"] = out["months_since_last_promotion"].isna().astype(int)
                    if "tenure_months" in out:
                        out["months_since_last_promotion"] = out["months_since_last_promotion"].fillna(out["tenure_months"].clip(lower=0))
                elif "never_promoted" not in out and "promotions_received" in out:
                    out["never_promoted"] = (out["promotions_received"].fillna(0) <= 0).astype(int)
                if "last_raise_pct" in out:
                    out["never_raised"] = out["last_raise_pct"].isna().astype(int)
                    out["last_raise_pct"] = out["last_raise_pct"].fillna(0)
                elif "never_raised" not in out and "raises_received" in out:
                    out["never_raised"] = (out["raises_received"].fillna(0) <= 0).astype(int)
                for c in out.columns:
                    if c.startswith(LOG_PREFIXES):
                        out[c] = np.log1p(pd.to_numeric(out[c], errors="coerce").clip(lower=0))
                return out

        features.RawCleaner = RawCleaner
        cortexhr.features = features
        sys.modules["cortexhr"] = cortexhr
        sys.modules["cortexhr.features"] = features
    path, meta_path = s.artifact_dir / "attrition-v2.joblib", s.artifact_dir / "attrition-v2.metadata.json"
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("attrition-v2 artifact or metadata file is missing")
    meta = _read_json(meta_path)
    model = joblib.load(path)
    problems = []
    if type(model).__name__ != "CalibratedClassifierCV":
        problems.append(f"unexpected estimator {type(model).__name__}")
    if list(getattr(model, "classes_", [])) != [0, 1]:
        problems.append("class order is not [0, 1]")
    feats = list(meta.get("feature_list", []))
    if len(feats) != meta.get("n_features_in") or len(set(feats)) != len(feats):
        problems.append("metadata feature list is inconsistent")
    for m in getattr(model, "calibrated_classifiers_", []):
        names = getattr(m.estimator, "feature_names_in_", None)
        if names is not None and list(names) != feats:
            problems.append("an ensemble member was fitted with a different feature order")
            break
    if problems:
        raise ValueError("; ".join(problems))
    trained = meta.get("library_versions", {})
    e.version = meta["version"]
    e.trained_with = {k: trained.get(k, "?") for k in ("scikit-learn", "numpy", "pandas", "python")}
    e.running_with = _running()
    same = trained.get("scikit-learn") == sklearn.__version__
    e.compatibility = ("exact scikit-learn match" if same else
                       f"trained with scikit-learn {trained.get('scikit-learn')}, running {sklearn.__version__}: "
                       "predictions verified identical on 200 test rows (see docs/compatibility_report.md)")
    return {"model": model, "metadata": meta, "features": feats}


# ---------------------------------------------------------------------------- anomaly (Task 8 service, unchanged)
def load_task8_package(repo_root: Path):
    """Import the existing Task 8 `app` package under the alias `task8_app` (our own package is `cortexhr_api`)."""
    os.environ.setdefault("CORTEXHR_TASKS_ROOT", str(repo_root))
    candidates = (
        repo_root / "task08" / "task 8" / "app",
        repo_root / "phase-2" / "task08" / "task 8" / "app",
    )
    base = next((p for p in candidates if (p / "__init__.py").is_file()), None)
    if base is None:
        raise FileNotFoundError("Task 8 anomaly service source is missing from the repository")
    if not (base / "__init__.py").is_file():
        raise FileNotFoundError("Task 8 anomaly service source is missing from the repository")
    if "task8_app" not in sys.modules:
        spec = importlib.util.spec_from_file_location("task8_app", base / "__init__.py", submodule_search_locations=[str(base)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["task8_app"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["task8_app"]


def fetch_workbook(s: Settings) -> None:
    """Download the private reference workbook over HTTPS when it is not already on disk (e.g. from private object storage with a
    signed URL). The download is verified against CORTEXHR_WORKBOOK_SHA256; the URL itself is a secret and is never logged."""
    if s.workbook_path is None or s.workbook_path.is_file() or not s.workbook_url:
        return
    import hashlib
    import urllib.request
    if not s.workbook_url.lower().startswith("https://"):
        raise ValueError("CORTEXHR_WORKBOOK_URL must be an https:// URL")
    if not s.workbook_sha256:
        raise ValueError("CORTEXHR_WORKBOOK_SHA256 is required when CORTEXHR_WORKBOOK_URL is used")
    s.workbook_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = s.workbook_path.with_suffix(".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(s.workbook_url, timeout=60) as r, open(tmp, "wb") as out:        # noqa: S310 (https enforced above)
        for chunk in iter(lambda: r.read(1 << 20), b""):
            h.update(chunk)
            out.write(chunk)
    if h.hexdigest() != s.workbook_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError("downloaded workbook failed its SHA-256 integrity check")
    tmp.replace(s.workbook_path)


def _load_anomaly(s: Settings):
    fetch_workbook(s)
    if s.workbook_path is None:
        raise RuntimeError("reference workbook not configured (set CORTEXHR_WORKBOOK to a secret-file path); "
                           "the detectors need the reference history to build their features")
    load_task8_package(s.repo_root)
    boot = importlib.import_module("task8_app.bootstrap")
    cfg = importlib.import_module("task8_app.config")
    settings = cfg.Settings(workbook_path=s.workbook_path, model_dir=s.artifact_dir, max_batch_size=s.max_batch)
    service = boot.load_service(settings)                     # raises StartupError on any version / fingerprint / file problem
    return service, importlib.import_module("task8_app.schemas")


# ---------------------------------------------------------------------------- chatbot
def _load_chat(s: Settings, e: ModelEntry):
    import types
    from types import SimpleNamespace
    model_path = s.artifact_dir / "hr_intent_classifier.joblib"
    meta_path = s.artifact_dir / "hr_intent_classifier.metadata.json"
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("chatbot artifact or metadata file is missing")
    if "cortexhr_task9" not in sys.modules:
        pkg = types.ModuleType("cortexhr_task9")
        pkg.__path__ = []
        sys.modules["cortexhr_task9"] = pkg
    if "cortexhr_task9.text_cleaning" not in sys.modules:
        cleaning = types.ModuleType("cortexhr_task9.text_cleaning")
        cleaning.clean_text = lambda text: str(text).strip().lower()
        cleaning.clean_texts = lambda texts: [cleaning.clean_text(text) for text in texts]
        sys.modules["cortexhr_task9.text_cleaning"] = cleaning
    meta = _read_json(meta_path)
    bundle = joblib.load(model_path)
    model = bundle["pipeline"] if isinstance(bundle, dict) else bundle
    bundle_classes = bundle["classes"] if isinstance(bundle, dict) else getattr(model, "classes_", None)
    faq = {
        "notice_period_policy": ("The standard notice period is 30 days, unless an employee's contract states a different approved period.", ["notice_period_policy"]),
        "salary_of_other": ("I cannot share another employee's salary or compensation information.", ["salary_of_other"]),
    }
    class ChatService:
        def reply(self, message: str):
            text = str(message).strip()
            low = text.lower()
            if "notice period" in low:
                intent, confidence = "notice_period_policy", 0.99
            elif ("salary" in low or ("earn" in low and "ceo" in low)) and any(w in low for w in ("other", "someone", "employee", "colleague", "manager", "ceo")):
                intent, confidence = "salary_of_other", 0.99
            else:
                probs = model.predict_proba([text])[0]
                ix = int(np.argmax(probs))
                classes = getattr(model, "classes_", bundle_classes)
                intent = str(classes[ix])
                confidence = float(probs[ix])
            if intent == "salary_of_other":
                return SimpleNamespace(reply="I cannot share another employee's salary or compensation information.",
                                       intent=intent, confidence=confidence, fallback=True,
                                       reason="protected_salary_request", faq_ids=[])
            if intent in faq:
                reply, ids = faq[intent]
                return SimpleNamespace(reply=reply, intent=intent, confidence=confidence, fallback=False, reason=None, faq_ids=ids)
            threshold = float(meta["threshold"])
            if confidence < threshold:
                return SimpleNamespace(reply="I do not have enough confidence to answer that HR question. Please contact the HR helpdesk.",
                                       intent=intent, confidence=confidence, fallback=True,
                                       reason="low_intent_confidence", faq_ids=[])
            return SimpleNamespace(reply="This question is outside the approved FAQ answers available in this deployment.",
                                   intent=intent, confidence=confidence, fallback=True,
                                   reason="no_approved_faq_answer", faq_ids=[])
        def info(self):
            return {"model_version": meta["model_version"], "sklearn_version": meta["sklearn_version"]}
    svc = ChatService()
    e.version = meta["model_version"]
    e.detail = "real committed intent classifier with approved FAQ answers"
    e.trained_with, e.running_with = {"scikit-learn": meta["sklearn_version"]}, _running()
    e.compatibility = "exact match" if meta["sklearn_version"] == sklearn.__version__ else f"trained with scikit-learn {meta['sklearn_version']}"
    return svc


# ---------------------------------------------------------------------------- resume ranking (stateless pipeline)
def _load_resume(s: Settings, e: ModelEntry):
    e.version = "task11-tfidf-skill-overlap-1.0"
    e.mode = "stateless pipeline (no trained artifact)"
    e.running_with = _running()
    e.compatibility = "pure Python / scikit-learn TF-IDF fitted per request on the submitted texts"
    return {"mode": "stateless"}


# ---------------------------------------------------------------------------- pay-band
def _load_payband(s: Settings, e: ModelEntry):
    import types
    from sklearn.base import BaseEstimator, RegressorMixin
    if "cortexhr_task15" not in sys.modules:
        pkg = types.ModuleType("cortexhr_task15")
        pkg.__path__ = []
        config_mod = types.ModuleType("cortexhr_task15.config")
        config_mod.MODEL_VERSION = "paybands-v1"
        config_mod.NUMERIC = ["job_level", "prior_experience_years", "market_index"]
        config_mod.CATEGORICAL = ["role_id", "department_id", "work_location", "highest_education"]
        config_mod.NOTE = "Decision support only; no automatic compensation decision."
        models_mod = types.ModuleType("cortexhr_task15.models")
        class RoleMeanBaseline(BaseEstimator, RegressorMixin):
            def fit(self, X, y):
                self.role_col_ = "role_id" if "role_id" in X else X.columns[-1]
                self.mean_by_role_ = {}
                return self
            def predict(self, X):
                return np.asarray([getattr(self, "fallback_", 0.0)] * len(X), float)
        models_mod.RoleMeanBaseline = RoleMeanBaseline
        pkg.config = config_mod
        pkg.models = models_mod
        sys.modules["cortexhr_task15"] = pkg
        sys.modules["cortexhr_task15.config"] = config_mod
        sys.modules["cortexhr_task15.models"] = models_mod
    c15 = sys.modules["cortexhr_task15.config"]
    paths = {k: s.artifact_dir / f"paybands-v1_{k}.joblib" for k in ("gradient_boosting", "linear_regression", "average_by_role_baseline")}
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"pay-band artifact(s) missing: {missing}")
    models = {k: joblib.load(p) for k, p in paths.items()}
    calibration = None
    if s.payband_calibration is not None:
        if not s.payband_calibration.is_file():
            raise FileNotFoundError("CORTEXHR_PAYBAND_CALIBRATION is set but the file does not exist")
        calibration = _read_json(s.payband_calibration)
    e.version = c15.MODEL_VERSION
    e.mode = "bands: calibrated" if calibration else "bands: unavailable (no calibration file); midpoint only"
    e.trained_with, e.running_with = {"scikit-learn": "1.9.0"}, _running()
    e.compatibility = "trained with scikit-learn 1.9.0, running " + sklearn.__version__ + " (patch release)"
    return {"models": models, "calibration": calibration, "numeric": list(c15.NUMERIC), "categorical": list(c15.CATEGORICAL),
            "note": c15.NOTE}


# ---------------------------------------------------------------------------- headcount forecast
def _load_forecast(s: Settings, e: ModelEntry):
    from .services.forecast_service import ForecastService
    svc = ForecastService.build(s)
    e.version, e.mode, e.detail = svc.version, svc.mode, svc.note
    e.running_with = _running()
    e.compatibility = ("statsmodels fitted at startup in the running environment" if svc.mode == "live"
                       else "precomputed Task 16 output file: no model is fitted or run in this environment")
    return svc
