"""Load the REAL attrition-v2 artifact created in Task 13 (no training, no replacement) and verify it before use."""
from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
import sklearn

from . import config

config.ensure_task13_importable()
from attrition_v2 import model_versioning as mv  # noqa: E402  (Task 13 package: load_model / predict_attrition / build_scoring_table)


class ModelVerificationError(RuntimeError):
    pass


def load_attrition_v2():
    """Task 13's own loader. Refuses any version other than attrition-v2."""
    loaded = mv.load_model(config.MODEL_VERSION)
    verify_model(loaded)
    return loaded


def verify_model(loaded) -> Dict[str, Any]:
    """Checks that this really is attrition-v2: version, estimator type, class order, ensemble size, feature order."""
    md, model = loaded.metadata, loaded.model
    problems = []
    if loaded.version != config.MODEL_VERSION or md.get("version") != config.MODEL_VERSION:
        problems.append(f"model version is {loaded.version!r}, expected {config.MODEL_VERSION!r}")
    if type(model).__name__ != "CalibratedClassifierCV":
        problems.append(f"estimator is {type(model).__name__}, expected CalibratedClassifierCV")
    if list(getattr(model, "classes_", [])) != [0, 1]:
        problems.append(f"class order {list(getattr(model, 'classes_', []))} != [0, 1]; column 1 would not be the attrition probability")
    members = getattr(model, "calibrated_classifiers_", [])
    if not members:
        problems.append("no calibrated members found")
    feats = list(md.get("feature_list", []))
    if len(feats) != md.get("n_features_in") or len(feats) != len(set(feats)):
        problems.append("metadata feature list is inconsistent")
    for i, m in enumerate(members):
        names = getattr(m.estimator, "feature_names_in_", None)
        if names is not None and list(names) != feats:
            problems.append(f"member {i} was fitted with a different feature order")
    trained = md.get("library_versions", {}).get("scikit-learn")
    if trained and trained != sklearn.__version__:
        warnings.warn(f"attrition-v2 was trained with scikit-learn {trained}; running {sklearn.__version__}", RuntimeWarning)
    if problems:
        raise ModelVerificationError("; ".join(problems))
    return {"model_version": loaded.version, "estimator": type(model).__name__, "classes": [0, 1], "ensemble_members": len(members),
            "n_features": len(feats), "feature_list": feats, "calibration": md.get("calibration_method"),
            "classifier": md.get("hyperparameters", {}).get("classifier"), "trained_at_utc": md.get("trained_at_utc"),
            "trained_with_sklearn": trained, "running_sklearn": sklearn.__version__}
