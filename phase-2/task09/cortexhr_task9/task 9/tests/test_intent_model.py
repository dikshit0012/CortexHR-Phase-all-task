import inspect
import json

import joblib
import numpy as np
import pytest

from cortexhr_task9 import intent_model, training
from cortexhr_task9.data import load_training, sha256_file
from cortexhr_task9.intent_model import ModelLoadError, load_artifact
from cortexhr_task9.paths import TEST_CSV, TRAIN_CSV, TASK9_ROOT


def test_artifact_can_be_loaded_and_is_one_pipeline(model):
    assert model.version == "hr-intent-v1"
    assert len(model.classes) == 33
    assert [name for name, _ in model.pipeline.steps] == ["clean", "tfidf", "clf"]      # cleaning + TF-IDF + classifier together
    assert 0.0 < model.threshold < 1.0
    meta = model.metadata
    import sklearn
    assert meta["sklearn_version"] == sklearn.__version__
    assert meta["training"]["rows"] == 858 and meta["training"]["intents"] == 33


def test_model_classifies_known_hr_questions(model):
    for text, expected in [("what is the work from home policy", "wfh_policy"),
                           ("plesae what is the notice period", "notice_period_policy"),
                           ("how is pf calculated", "pf_info"),
                           ("when will i get form 16", "tax_declaration")]:
        pred = model.predict(text)
        assert pred.intent == expected, (text, pred)
        assert pred.confidence >= model.threshold, (text, pred)


def test_probabilities_are_a_valid_distribution(model):
    proba = model.predict_proba(["leave balance", "asdfgh", "who do i report to"])
    assert proba.shape == (3, 33)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_training_is_deterministic(artifact_path, model):
    again = train_again(artifact_path.parent / "second.joblib")
    texts = ["how many casual leaves", "show my payslip", "tell me a joke"]
    assert np.allclose(again.predict_proba(texts), model.predict_proba(texts))
    assert again.threshold == model.threshold


def train_again(path):
    training.train_and_save(TRAIN_CSV, path, report_dir=None, run_comparison=False)
    return load_artifact(path)


def test_official_test_set_is_not_used_for_training_or_threshold_selection(model):
    assert model.metadata["training"]["data_sha256"] == sha256_file(TRAIN_CSV)
    assert model.metadata["training"]["data_sha256"] != sha256_file(TEST_CSV)
    assert model.metadata["training"]["rows"] == len(load_training(TRAIN_CSV))
    assert "official test set not used" in model.metadata["threshold_selection"]["method"]
    for module in (training, intent_model):                                    # the training code never even names the test file
        src = inspect.getsource(module)
        assert "chatbot_test_set" not in src and "TEST_CSV" not in src
    script = (TASK9_ROOT / "scripts" / "train_intent_model.py").read_text(encoding="utf-8")
    assert "TEST_CSV" not in script and "load_test" not in script


def test_missing_artifact_is_reported_clearly(tmp_path):
    with pytest.raises(ModelLoadError) as err:
        load_artifact(tmp_path / "nope.joblib")
    assert "missing" in str(err.value) and str(tmp_path) not in str(err.value)       # no absolute path in the message


def test_corrupted_artifact_is_refused(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"this is not a joblib file")
    with pytest.raises(ModelLoadError, match="could not be loaded"):
        load_artifact(bad)


def _tampered(artifact_path, tmp_path, **changes):
    bundle = joblib.load(artifact_path)
    bundle.update(changes)
    out = tmp_path / "tampered.joblib"
    joblib.dump(bundle, out)
    return out


@pytest.mark.parametrize("change,message", [
    ({"sklearn_version": "0.0.1"}, "scikit-learn"),
    ({"numpy_version": "0.0.1"}, "NumPy"),
    ({"cleaning_version": "clean-v0"}, "text cleaning"),
    ({"format_version": 99}, "format"),
    ({"threshold": 1.5}, "threshold"),
])
def test_incompatible_artifacts_are_refused_with_a_clear_message(artifact_path, tmp_path, change, message):
    with pytest.raises(ModelLoadError, match=message):
        load_artifact(_tampered(artifact_path, tmp_path, **change))


def test_incomplete_artifact_is_refused(artifact_path, tmp_path):
    bundle = joblib.load(artifact_path)
    del bundle["classes"]
    out = tmp_path / "incomplete.joblib"
    joblib.dump(bundle, out)
    with pytest.raises(ModelLoadError, match="incomplete"):
        load_artifact(out)


def test_metadata_json_is_written_and_contains_no_training_text(artifact_path):
    meta = json.loads(artifact_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert "pipeline" not in meta and meta["threshold"] == load_artifact(artifact_path).threshold
    assert "show my payslip" not in json.dumps(meta)
