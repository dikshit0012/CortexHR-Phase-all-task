from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from app import app as vercel_app
from fastapi.testclient import TestClient

_task16_app_path = Path(__file__).resolve().parents[1] / "app.py"
_task16_spec = spec_from_file_location("task16_api_under_test", _task16_app_path)
if _task16_spec is None or _task16_spec.loader is None:
    raise ImportError(f"Unable to load Task 16 API at {_task16_app_path}")

task16_app = module_from_spec(_task16_spec)
_task16_spec.loader.exec_module(task16_app)
client = TestClient(task16_app.app)
vercel_client = TestClient(vercel_app)


def test_health_endpoint():
    response = vercel_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_homepage_serves_forecast_dashboard():
    response = vercel_client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Workforce outlook" in response.text
    assert 'id="forecast-table"' in response.text
    assert 'fetch("/forecast")' in response.text


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
