import app as task16_app
from fastapi.testclient import TestClient

client = TestClient(task16_app.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_forecast_endpoint_returns_generated_rows(tmp_path, monkeypatch):
    forecast_path = tmp_path / "headcount_forecast.csv"
    forecast_path.write_text(
        "forecast_id,metric,department_id,forecast_month,forecast_value\n"
        "FC-0001,headcount_end,ALL,2026-10-01,444.4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(task16_app, "FORECAST_PATH", forecast_path)

    response = client.get("/forecast")

    assert response.status_code == 200
    assert response.json()["forecasts"] == [
        {
            "forecast_id": "FC-0001",
            "metric": "headcount_end",
            "department_id": "ALL",
            "forecast_month": "2026-10-01",
            "forecast_value": 444.4,
        }
    ]


def test_forecast_endpoint_reports_missing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        task16_app, "FORECAST_PATH", tmp_path / "missing_forecast.csv"
    )

    response = client.get("/forecast")

    assert response.status_code == 503
    assert "Run scripts/run_task16.py" in response.json()["detail"]
