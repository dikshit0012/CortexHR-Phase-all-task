import pytest

from conftest import gross_above_norm_payroll

GENERIC_500 = {"detail": "Internal error while scoring the request."}


def test_prediction_failure_returns_generic_500_without_internals(client, service, make_attendance, monkeypatch):
    def broken(_feats):
        raise RuntimeError("secret detail EMP-9999 /private/path/model.joblib")
    monkeypatch.setattr(service.attendance.detector, "predict", broken)
    r = client.post("/v1/anomaly/attendance", json=make_attendance())
    assert r.status_code == 500 and r.json() == GENERIC_500
    assert "secret" not in r.text and "EMP-9999" not in r.text and "Traceback" not in r.text


def test_preprocessing_failure_returns_generic_500(client, make_payroll, monkeypatch):
    import app.scoring as scoring

    def broken(_df):
        raise ValueError("bad value for EMP-0001 in column gross_pay")
    monkeypatch.setattr(scoring, "build_payroll_features", broken)
    r = client.post("/v1/anomaly/payroll", json=make_payroll())
    assert r.status_code == 500 and r.json() == GENERIC_500
    assert "EMP-0001" not in r.text


def test_failure_in_a_batch_is_500_not_a_partial_result(client, service, make_payroll, monkeypatch):
    monkeypatch.setattr(service.payroll.detector, "predict", lambda f: (_ for _ in ()).throw(RuntimeError("x")))
    r = client.post("/v1/anomaly/payroll/batch", json={"records": [make_payroll(), make_payroll("PI-2", "EMP-0002")]})
    assert r.status_code == 500 and "results" not in r.json()


def test_validation_errors_do_not_echo_submitted_values(client, make_attendance):
    body = make_attendance(employee_id="EMP-SECRET-77", present_days=-5)
    r = client.post("/v1/anomaly/attendance", json=body)
    assert r.status_code == 422
    assert "EMP-SECRET-77" not in r.text and '"input"' not in r.text
    assert set(r.json()["detail"][0]) == {"loc", "type", "msg"}


def test_service_still_works_after_a_failed_request(client, service, make_attendance, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(service.attendance.detector, "predict", lambda f: (_ for _ in ()).throw(RuntimeError("x")))
        assert client.post("/v1/anomaly/attendance", json=make_attendance()).status_code == 500
    assert client.post("/v1/anomaly/attendance", json=make_attendance()).status_code == 200


def test_unknown_route_is_404(client):
    assert client.get("/v1/does-not-exist").status_code == 404
