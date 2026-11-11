"""Intent classifier: build (offline), save / load (versioned artifact) and predict.

The saved object is ONE scikit-learn Pipeline: clean_texts -> TF-IDF (word 1-2 grams + character 2-5 grams) -> LogisticRegression.
Cleaning is inside the pipeline, so inference applies exactly the training-time preprocessing.

SECURITY: joblib uses pickle, so loading an artifact can run code. Only load artifacts built by this project.
"""
from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from .text_cleaning import CLEANING_VERSION, clean_texts

FORMAT_VERSION = 1
MODEL_VERSION = "hr-intent-v1"
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.60            # the brief's starting value, used only if no threshold could be selected
# libraries whose version changes how a pickled pipeline loads / predicts: a mismatch is refused
_STRICT = (("scikit-learn", "sklearn_version", sklearn.__version__), ("NumPy", "numpy_version", np.__version__),
           ("SciPy", "scipy_version", scipy.__version__), ("joblib", "joblib_version", joblib.__version__))
_REQUIRED_KEYS = ("format_version", "model_version", "cleaning_version", "pipeline", "classes", "threshold",
                  "threshold_selection", "sklearn_version", "numpy_version", "scipy_version", "joblib_version",
                  "python_version", "created_at", "training", "config")


class ModelLoadError(RuntimeError):
    """The artifact is missing, corrupted, incompatible or incomplete. Messages never contain absolute paths."""


@dataclass(frozen=True)
class ModelConfig:
    word_ngrams: tuple = (1, 2)
    char_ngrams: Optional[tuple] = (2, 5)       # None -> word features only (the brief's baseline)
    C: float = 10.0
    min_df: int = 1

    def as_dict(self) -> Dict[str, Any]:
        return {"word_ngrams": list(self.word_ngrams), "char_ngrams": list(self.char_ngrams) if self.char_ngrams else None,
                "C": self.C, "min_df": self.min_df, "class_weight": "balanced", "max_iter": 2000,
                "random_state": RANDOM_STATE, "sublinear_tf": True}


# The brief's starting configuration (word n-grams only, C=1) - kept for the comparison table in the training report.
BASELINE = ModelConfig(char_ngrams=None, C=1.0)


def build_pipeline(config: ModelConfig = ModelConfig()) -> Pipeline:
    word = TfidfVectorizer(ngram_range=config.word_ngrams, min_df=config.min_df, sublinear_tf=True)
    if config.char_ngrams:
        features: Any = FeatureUnion([
            ("word", word),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=config.char_ngrams, min_df=config.min_df, sublinear_tf=True)),
        ])
    else:
        features = word
    return Pipeline([
        ("clean", FunctionTransformer(clean_texts, validate=False)),
        ("tfidf", features),
        ("clf", LogisticRegression(C=config.C, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def save_artifact(path: Path, pipeline: Pipeline, threshold: float, threshold_selection: Dict[str, Any],
                  training: Dict[str, Any], config: ModelConfig, created_at: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write the joblib artifact (atomically) and a human-readable metadata JSON next to it. Returns the metadata."""
    bundle = {
        "format_version": FORMAT_VERSION, "model_version": MODEL_VERSION, "cleaning_version": CLEANING_VERSION,
        "pipeline": pipeline, "classes": [str(c) for c in pipeline.classes_], "threshold": float(threshold),
        "threshold_selection": threshold_selection, "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__, "joblib_version": joblib.__version__,
        "python_version": platform.python_version(), "created_at": created_at, "training": training,
        "config": config.as_dict(), **(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".joblib.tmp")
    joblib.dump(bundle, tmp, compress=3)
    os.replace(tmp, path)                                      # never leave a half-written artifact
    meta = {k: v for k, v in bundle.items() if k != "pipeline"}
    path.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float          # highest predict_proba value: NOT a calibrated probability
    cleaned_text: str


@dataclass(frozen=True)
class LoadedIntentModel:
    pipeline: Pipeline
    classes: List[str]
    threshold: float
    metadata: Dict[str, Any]

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict(self, text: str) -> Prediction:
        """Top-1 intent and its probability for ONE message. Raises ValueError if the output is not a valid distribution."""
        proba = self.pipeline.predict_proba([text])[0]
        if proba.shape[0] != len(self.classes) or not np.all(np.isfinite(proba)):
            raise ValueError("invalid probability vector")
        i = int(np.argmax(proba))
        conf = float(proba[i])
        if not 0.0 <= conf <= 1.0 + 1e-9:
            raise ValueError("probability out of range")
        return Prediction(self.classes[i], min(conf, 1.0), clean_texts([text])[0])


def load_artifact(path: Path) -> LoadedIntentModel:
    """Load and verify. Every problem raises ModelLoadError with a plain-language fix hint (no dummy fallback)."""
    path = Path(path)
    hint = "Rebuild it with: python scripts/train_intent_model.py (from the `task 9` folder), in the same environment you serve from."
    if not path.is_file():
        raise ModelLoadError(f"Intent model artifact missing: {path.name}. {hint}")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"Intent model artifact {path.name} could not be loaded ({type(exc).__name__}); it may be corrupted or built with another library version. {hint}") from None
    if not isinstance(bundle, dict):
        raise ModelLoadError(f"Intent model artifact {path.name} has an unexpected structure. {hint}")
    missing = [k for k in _REQUIRED_KEYS if k not in bundle]
    if missing:
        raise ModelLoadError(f"Intent model artifact {path.name} is incomplete (missing {missing}). {hint}")
    if bundle["format_version"] != FORMAT_VERSION or bundle["model_version"] != MODEL_VERSION:
        raise ModelLoadError(f"Intent model artifact {path.name} is {bundle['model_version']} (format {bundle['format_version']}), "
                             f"this code needs {MODEL_VERSION} (format {FORMAT_VERSION}). {hint}")
    if bundle["cleaning_version"] != CLEANING_VERSION:
        raise ModelLoadError(f"Intent model artifact {path.name} was trained with text cleaning {bundle['cleaning_version']}, "
                             f"this code has {CLEANING_VERSION}. Training and inference preprocessing must match. {hint}")
    for library, key, installed in _STRICT:
        if bundle[key] != installed:
            raise ModelLoadError(f"Intent model artifact {path.name} was built with {library} {bundle[key]} but this environment "
                                 f"has {installed}. {hint}")
    pipe = bundle["pipeline"]
    if not (hasattr(pipe, "predict_proba") and hasattr(pipe, "classes_")):
        raise ModelLoadError(f"Intent model artifact {path.name} does not contain a usable classifier. {hint}")
    classes = [str(c) for c in pipe.classes_]
    if classes != bundle["classes"] or len(set(classes)) != len(classes) or not classes:
        raise ModelLoadError(f"Intent model artifact {path.name} has inconsistent class labels. {hint}")
    thr = bundle["threshold"]
    if not (isinstance(thr, float) and 0.0 < thr < 1.0):
        raise ModelLoadError(f"Intent model artifact {path.name} has an invalid confidence threshold. {hint}")
    meta = {k: v for k, v in bundle.items() if k != "pipeline"}
    return LoadedIntentModel(pipeline=pipe, classes=classes, threshold=thr, metadata=meta)
