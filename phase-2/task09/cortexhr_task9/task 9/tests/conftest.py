"""Shared fixtures. The model artifact used by most tests is BUILT INTO A TEMP FOLDER from the training CSV (so the tests do not
depend on a local models/ folder) and then loaded from disk, exactly like production does."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cortexhr_task9.chat_service import ChatService  # noqa: E402
from cortexhr_task9.faq import load_responder  # noqa: E402
from cortexhr_task9.intent_model import Prediction, load_artifact  # noqa: E402
from cortexhr_task9.paths import DEFAULT_FAQ, INTENT_FAQ_MAP, TRAIN_CSV  # noqa: E402
from cortexhr_task9.text_cleaning import clean_text  # noqa: E402
from cortexhr_task9.training import train_and_save  # noqa: E402


@pytest.fixture(scope="session")
def artifact_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("models") / "hr_intent_classifier.joblib"
    train_and_save(TRAIN_CSV, path, report_dir=None, run_comparison=False)
    return path


@pytest.fixture(scope="session")
def model(artifact_path):
    return load_artifact(artifact_path)


@pytest.fixture(scope="session")
def responder(model):
    return load_responder(DEFAULT_FAQ, INTENT_FAQ_MAP, model.classes)


@pytest.fixture(scope="session")
def service(model, responder) -> ChatService:
    return ChatService(model, responder)


class FakeModel:
    """Deterministic stand-in for the loaded classifier: always predicts `intent` with `confidence` (or raises `error`)."""
    version = "fake-model"
    classes = ["wfh_policy"]
    metadata = {"training": {"rows": 0}, "sklearn_version": "x", "created_at": "x"}

    def __init__(self, intent="wfh_policy", confidence=0.95, threshold=0.64, error=None):
        self.intent, self.confidence, self.threshold, self.error = intent, confidence, threshold, error

    def predict(self, text):
        if self.error:
            raise self.error
        return Prediction(self.intent, self.confidence, clean_text(text))


@pytest.fixture()
def fake_service(responder):
    def make(**kw):
        return ChatService(FakeModel(**kw), responder)
    return make
