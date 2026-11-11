import json
import shutil

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.bootstrap as bootstrap
from app.artifacts import ATTENDANCE, PAYROLL, VERSIONS, artifact_path, load_artifact
from app.config import ENV_MAX_BATCH, ENV_WORKBOOK, MAX_BATCH_CEILING, Settings, StartupError
from app.main import create_app
from app.reference import ATTENDANCE_SHEET, PAYROLL_SHEET, load_reference
from conftest import TEST_BATCH_LIMIT, gross_above_norm_payroll, make_reference_frames, write_workbook


# ---- Test 5: models load once at startup and are reused ---------------------------------------------------------
def test_models_are_loaded_once_at_startup_and_reused_across_requests(settings, make_attendance, make_payroll, monkeypatch):
    from cortexhr_task5.attendance_anomaly_detector import AttendanceAnomalyDetector
    from cortexhr_task6.payroll_model import PayrollAnomalyModel

    loads = []
    reference_loads = []
    real_load = bootstrap.load_artifact
    real_reference_load = bootstrap.load_reference
    monkeypatch.setattr(bootstrap, "load_artifact", lambda path, kind, ref: loads.append(kind) or real_load(path, kind, ref))
    monkeypatch.setattr(bootstrap, "load_reference",
                        lambda path: reference_loads.append(path) or real_reference_load(path))

    def refuse(*_a, **_k):
        raise AssertionError("a model was FITTED while the service was running")
    monkeypatch.setattr(AttendanceAnomalyDetector, "fit", refuse)       # fitting only happens in scripts/build_artifacts.py
    monkeypatch.setattr(PayrollAnomalyModel, "fit", refuse)

    application = create_app(settings=settings)
    assert loads == []                                                  # nothing is loaded at import / app creation
    assert reference_loads == []
    with TestClient(application) as c:
        assert sorted(loads) == sorted([ATTENDANCE, PAYROLL])           # exactly one load per artifact, at startup
        assert reference_loads == [settings.workbook_path]              # one reference load at startup
        svc = application.state.service
        before = (id(svc), id(svc.attendance.detector), id(svc.payroll.detector),
                  id(svc.attendance.detector.model_), id(svc.payroll.detector.model_))
        for i in range(3):
            assert c.post("/v1/anomaly/attendance", json=make_attendance(f"AF-R-{i}")).status_code == 200
            assert c.post("/v1/anomaly/payroll", json=make_payroll(f"PI-R-{i}")).status_code == 200
        assert c.post("/v1/anomaly/payroll/batch", json={"records": [make_payroll("PI-R-B1"), make_payroll("PI-R-B2", "EMP-0002")]}).status_code == 200
        after = (id(svc), id(svc.attendance.detector), id(svc.payroll.detector),
                 id(svc.attendance.detector.model_), id(svc.payroll.detector.model_))
        assert before == after                                         # the same objects served every request
        assert sorted(loads) == sorted([ATTENDANCE, PAYROLL])           # still no further loads
        assert reference_loads == [settings.workbook_path]              # reference history is never reloaded
    assert application.state.service is None                            # released at shutdown


def test_loaded_detectors_have_the_expected_interfaces(service):
    for loaded, version in ((service.attendance, VERSIONS[ATTENDANCE]), (service.payroll, VERSIONS[PAYROLL])):
        assert loaded.version == version
        assert callable(loaded.detector.predict)
        assert hasattr(loaded.detector.imputer_, "transform") and hasattr(loaded.detector.model_, "decision_function")
        assert loaded.detector.kept_features_
        assert loaded.scale.lo < loaded.scale.hi


def test_health_and_models_endpoints_expose_versions_but_no_paths(client, model_dir, workbook):
    h = client.get("/v1/health")
    assert h.status_code == 200 and h.json() == {"status": "ok", "service": "cortexhr-anomaly", "models_loaded": True}
    m = client.get("/v1/models")
    assert m.status_code == 200
    body = m.json()
    assert body["max_batch_size"] == TEST_BATCH_LIMIT
    assert body["detectors"]["attendance"]["artifact_version"] == "attendance-anomaly-v1"
    assert body["detectors"]["payroll"]["artifact_version"] == "payroll-anomaly-v1"
    assert set(body["detectors"]["attendance"]["dependency_versions"]) == {
        "python", "numpy", "pandas", "scipy", "sklearn", "joblib", "openpyxl"
    }
    text = m.text
    assert str(model_dir) not in text and str(workbook) not in text and ".joblib" not in text and ".xlsx" not in text


def test_service_without_loaded_models_answers_503(make_attendance):
    c = TestClient(create_app())                                        # lifespan NOT entered: nothing is loaded
    assert c.get("/v1/health").status_code == 503
    r = c.post("/v1/anomaly/attendance", json=make_attendance())
    assert r.status_code == 503 and "not loaded" in r.json()["detail"]


# ---- artifacts: missing / corrupted / incompatible -> a clear startup error, never a dummy model --------------------
@pytest.fixture()
def models_copy(model_dir, tmp_path):
    d = tmp_path / "models"
    shutil.copytree(model_dir, d)
    return d


def start(settings):
    with TestClient(create_app(settings=settings)):
        pass


def with_models(settings, d):
    return Settings(workbook_path=settings.workbook_path, model_dir=d, max_batch_size=settings.max_batch_size)


def test_missing_artifact_stops_startup_with_a_clear_message(settings, models_copy):
    artifact_path(models_copy, PAYROLL).unlink()
    with pytest.raises(StartupError, match=r"missing: payroll-anomaly-v1\.joblib.*build_artifacts"):
        start(with_models(settings, models_copy))


def test_corrupted_artifact_stops_startup(settings, models_copy):
    artifact_path(models_copy, ATTENDANCE).write_bytes(b"this is not a joblib file")
    with pytest.raises(StartupError, match="could not be loaded"):
        start(with_models(settings, models_copy))


def test_swapped_artifacts_are_detected(settings, models_copy):
    shutil.copy(artifact_path(models_copy, PAYROLL), artifact_path(models_copy, ATTENDANCE))
    with pytest.raises(StartupError, match="must not be mixed up"):
        start(with_models(settings, models_copy))


@pytest.mark.parametrize(
    ("version_key", "library"),
    [
        ("python_version", "Python"),
        ("numpy_version", "NumPy"),
        ("pandas_version", "pandas"),
        ("scipy_version", "SciPy"),
        ("sklearn_version", "scikit-learn"),
        ("joblib_version", "joblib"),
        ("openpyxl_version", "openpyxl"),
    ],
)
def test_artifact_from_another_library_version_is_refused(settings, models_copy, version_key, library):
    p = artifact_path(models_copy, ATTENDANCE)
    bundle = joblib.load(p)
    bundle[version_key] = "0.0.1"
    joblib.dump(bundle, p)
    with pytest.raises(StartupError, match=library):
        start(with_models(settings, models_copy))


def test_incomplete_artifact_is_refused(settings, models_copy):
    p = artifact_path(models_copy, PAYROLL)
    bundle = joblib.load(p)
    del bundle["imputer"]
    joblib.dump(bundle, p)
    with pytest.raises(StartupError, match="incomplete"):
        start(with_models(settings, models_copy))


def test_artifact_built_from_a_different_reference_is_refused(settings, tmp_path):
    att, pay = make_reference_frames(n_emp=41, seed=7)                  # one more employee -> different record ids
    other = tmp_path / "other.xlsx"
    write_workbook(other, att, pay)
    with pytest.raises(StartupError, match="different reference workbook"):
        start(Settings(workbook_path=other, model_dir=settings.model_dir, max_batch_size=5))


# ---- reference workbook problems -----------------------------------------------------------------------------------
def test_missing_workbook_setting(settings):
    with pytest.raises(StartupError, match=ENV_WORKBOOK):
        start(Settings(workbook_path=None, model_dir=settings.model_dir))


def test_workbook_path_that_does_not_exist(settings, tmp_path):
    with pytest.raises(StartupError, match="not found"):
        start(Settings(workbook_path=tmp_path / "nope.xlsx", model_dir=settings.model_dir))


def test_workbook_path_that_is_a_directory_or_wrong_type(settings, tmp_path):
    with pytest.raises(StartupError, match="file, not a directory"):
        start(Settings(workbook_path=tmp_path, model_dir=settings.model_dir))
    txt = tmp_path / "data.txt"
    txt.write_text("x")
    with pytest.raises(StartupError, match=r"\.xlsx"):
        start(Settings(workbook_path=txt, model_dir=settings.model_dir))


def test_workbook_that_is_not_really_an_xlsx(settings, tmp_path):
    fake = tmp_path / "fake.xlsx"
    fake.write_bytes(b"not a workbook")
    with pytest.raises(StartupError, match="could not be read"):
        start(Settings(workbook_path=fake, model_dir=settings.model_dir))


def test_workbook_missing_a_required_sheet(settings, frames, tmp_path):
    p = tmp_path / "partial.xlsx"
    write_workbook(p, None, None, sheets={ATTENDANCE_SHEET: frames[0]})
    with pytest.raises(StartupError, match="missing a required sheet"):
        start(Settings(workbook_path=p, model_dir=settings.model_dir))


def test_workbook_missing_a_required_column(settings, frames, tmp_path):
    p = tmp_path / "nocol.xlsx"
    write_workbook(p, frames[0], frames[1].drop(columns=["gross_pay"]))
    with pytest.raises(StartupError, match="failed validation"):
        start(Settings(workbook_path=p, model_dir=settings.model_dir))


def test_label_sheets_are_never_read(workbook, frames, tmp_path, monkeypatch):
    p = tmp_path / "withlabels.xlsx"
    write_workbook(p, None, None, sheets={ATTENDANCE_SHEET: frames[0], PAYROLL_SHEET: frames[1],
                                          "payroll_anomaly_labels": pd.DataFrame({"payroll_item_id": ["PI-000001"], "label": [1]})})
    seen = []
    real = pd.ExcelFile.parse
    monkeypatch.setattr(pd.ExcelFile, "parse", lambda self, name=0, *a, **k: seen.append(name) or real(self, name, *a, **k))
    load_reference(p)
    assert seen == [ATTENDANCE_SHEET, PAYROLL_SHEET]                    # the label sheet is never opened


# ---- settings ------------------------------------------------------------------------------------------------------
def test_settings_from_environment():
    s = Settings.from_env({ENV_WORKBOOK: "x.xlsx", ENV_MAX_BATCH: "25"})
    assert s.max_batch_size == 25 and str(s.workbook_path) == "x.xlsx"
    assert Settings.from_env({}).max_batch_size == 100                  # documented default


@pytest.mark.parametrize("bad", ["abc", "0", "-3", str(MAX_BATCH_CEILING + 1), "2.5"])
def test_invalid_batch_limit_setting_stops_startup(bad):
    with pytest.raises(StartupError, match=ENV_MAX_BATCH):
        Settings.from_env({ENV_MAX_BATCH: bad})
