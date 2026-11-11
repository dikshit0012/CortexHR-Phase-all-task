"""Phase 10 + 11: save / load / use the versioned artifact ``attrition-v2`` (full preprocessing + model in one file).

    model_artifacts/attrition-v2/model.joblib   fitted estimator: Task 2 feature pipeline + classifier (+ calibration)
                                 metadata.json  what it is, how it was trained, which columns it needs
                                 metrics.json   honest (cross-validated) evidence for the decision

Rules: an existing version is never overwritten, nothing is retrained at load time, and prediction never prints or logs
employee values (only counts).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config
from cortexhr.features import check_no_forbidden

log = logging.getLogger("attrition_v2")


class ModelInputError(ValueError):
    """The records passed to ``predict_attrition`` do not match what the saved model was trained on."""


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_json(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=_jsonable, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# save
# ----------------------------------------------------------------------------------------------
def save_model(model, metadata: dict, metrics: dict, version: str = config.MODEL_VERSION, root=None) -> Path:
    """Write the three files. Refuses to overwrite an existing version (create a new version id instead)."""
    root = Path(root or config.ARTIFACTS)
    folder = root / version
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"{folder} already exists. Versions are never overwritten; choose a new version id.")
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata)
    meta.update({"version": version, "model_name": meta.get("model_name", "CortexHR attrition model"),
                 "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                                      "numpy": np.__version__, "pandas": pd.__version__,
                                      **_optional_versions()}})
    joblib.dump(model, folder / "model.joblib")
    write_json(meta, folder / "metadata.json")
    write_json(metrics, folder / "metrics.json")
    return folder


def _optional_versions() -> dict:
    out = {}
    try:
        import lightgbm
        out["lightgbm"] = lightgbm.__version__
    except Exception:       # pragma: no cover
        pass
    return out


# ----------------------------------------------------------------------------------------------
# load + predict
# ----------------------------------------------------------------------------------------------
class LoadedModel:
    def __init__(self, model, metadata: dict, metrics: dict, folder: Path):
        self.model, self.metadata, self.metrics, self.folder = model, metadata, metrics, folder

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @property
    def required_columns(self) -> list[str]:
        return list(self.metadata["feature_list"])


def load_model(version: str = config.MODEL_VERSION, root=None) -> LoadedModel:
    folder = Path(root or config.ARTIFACTS) / version
    for name in ("model.joblib", "metadata.json", "metrics.json"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing {folder / name}")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("version") != version:
        raise ValueError(f"Artifact folder says {version!r} but metadata says {metadata.get('version')!r}.")
    return LoadedModel(joblib.load(folder / "model.joblib"), metadata,
                       json.loads((folder / "metrics.json").read_text(encoding="utf-8")), folder)


def validate_input(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Check the records before scoring. Missing VALUES are fine (the Task 2 pipeline imputes them); missing or forbidden
    COLUMNS are errors. Returns the columns in the training order."""
    if not isinstance(df, pd.DataFrame):
        raise ModelInputError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if len(df) == 0:
        raise ModelInputError("No employee records to score.")
    check_no_forbidden(list(df.columns), "records to score")        # target / exit columns / identifiers
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ModelInputError(f"Missing required column(s): {missing}")
    return df[required].copy()


def predict_attrition(df: pd.DataFrame, loaded: LoadedModel | None = None) -> pd.DataFrame:
    """Probability of leaving within 12 months for each record, plus the model version. Never logs employee values."""
    loaded = loaded or load_model()
    X = validate_input(df, loaded.required_columns)
    p = loaded.model.predict_proba(X)[:, 1]
    if p.shape != (len(X),) or np.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise RuntimeError("The model produced invalid probabilities.")      # never expected; fail loudly
    log.info("scored %d records with %s", len(X), loaded.version)
    out = pd.DataFrame({"attrition_probability_12m": p, "model_version": loaded.version}, index=df.index)
    return out


def build_scoring_table(path=None) -> pd.DataFrame:
    """Real 'new employee' records: the ``attrition_scoring_2026`` sheet (no label) + the same Task 2 monthly features."""
    from cortexhr.features import select_features
    from cortexhr.monthly import add_monthly_features, load_monthly_tables
    path = path or config.WORKBOOK
    scoring = pd.read_excel(path, sheet_name="attrition_scoring_2026")
    attendance, payroll = load_monthly_tables(path)
    table, _ = add_monthly_features(scoring, attendance, payroll)
    return select_features(table)
