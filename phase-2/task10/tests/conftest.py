import pytest

from support import ScriptedClassifier, make_env


@pytest.fixture(scope="session")
def real_classifier():
    try:
        from cortexhr_task10.classifier import Task9Classifier
        return Task9Classifier.load()
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"Task 9 model could not be loaded: {type(exc).__name__}")


@pytest.fixture
def env_real(tmp_path, real_classifier):
    return make_env(tmp_path, real_classifier)


@pytest.fixture
def mk(tmp_path):
    def _mk(script=None, **kw):
        return make_env(tmp_path, ScriptedClassifier(script), **kw)
    return _mk
