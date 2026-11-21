import os
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import HEADCOUNT_CSV, make_client
from tests.samples import FORECAST

pytestmark = pytest.mark.real_model
COMMITTED = Path(__file__).resolve().parents[2] / "task16" / "outputs" / "headcount_forecast.csv"


def test_precomputed_mode_is_clearly_labelled(headers):
    with make_client(headcount_csv=None) as c:
        b = c.post("/v1/forecast/headcount", json=FORECAST, headers=headers).json()
        assert b["mode"] == "precomputed" and "PRECOMPUTED" in b["note"] and b["trained_through"] == "2026-09"
        ref = pd.read_csv(COMMITTED)
        ref = ref[(ref.metric == "headcount_end") & (ref.department_id == "ALL")]
        assert [p["expected_headcount"] for p in b["forecast"]] == list(ref.forecast_value)
        assert c.get("/v1/health").json()["forecast_mode"] == "precomputed"


@pytest.mark.skipif(not HEADCOUNT_CSV, reason="needs CORTEXHR_HEADCOUNT_CSV (private history file)")
def test_live_forecast_reproduces_task16_output(client, headers):
    b = client.post("/v1/forecast/headcount", json=FORECAST, headers=headers).json()
    assert b["mode"] == "live"
    ref = pd.read_csv(COMMITTED)
    for dept in ("ALL", "D01", "D05"):
        got = client.post("/v1/forecast/headcount", json={"department_id": dept, "horizon_months": 6}, headers=headers).json()["forecast"]
        exp = ref[(ref.metric == "headcount_end") & (ref.department_id == dept)]
        assert [p["expected_headcount"] for p in got] == list(exp.forecast_value), dept
        assert [p["lower_80"] for p in got] == list(exp.lower_80) and [p["upper_80"] for p in got] == list(exp.upper_80)


@pytest.mark.skipif(not HEADCOUNT_CSV, reason="needs CORTEXHR_HEADCOUNT_CSV")
def test_interval_ordering_horizon_and_events(client, headers):
    b = client.post("/v1/forecast/headcount", json={"department_id": "ALL", "horizon_months": 3}, headers=headers).json()
    assert len(b["forecast"]) == 3 and b["interval"] == 0.8
    assert all(p["lower_80"] <= p["expected_headcount"] <= p["upper_80"] for p in b["forecast"])
    assert any(e["event_id"] == "EV-01" for e in b["event_adjustments"]) and "no future events" in b["event_adjustment_note"]


def test_validation_and_errors(client, headers):
    assert client.post("/v1/forecast/headcount", json={"horizon_months": 7}, headers=headers).status_code == 422
    assert client.post("/v1/forecast/headcount", json={"horizon_months": 0}, headers=headers).status_code == 422
    assert client.post("/v1/forecast/headcount", json={"department_id": "bad id!"}, headers=headers).status_code == 422
    r = client.post("/v1/forecast/headcount", json={"department_id": "D99"}, headers=headers)
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_department"
    assert client.post("/v1/forecast/headcount", json=FORECAST).status_code == 401
