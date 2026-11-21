import numpy as np
import pandas as pd
import pytest

from cortexhr_task16 import event_handler as eh
from cortexhr_task16 import preprocessing as pp
from cortexhr_task16.data_loader import DataError


def test_classification_rules():
    assert eh.classify_event("COVID-19 hiring freeze") == "hiring_freeze"
    assert eh.classify_event("Data & AI team build-out") == "hiring_drive"
    assert eh.classify_event("Support scale-up for festive season") == "hiring_drive"
    assert eh.classify_event("Hybrid work policy introduced", "Three-day office policy") == eh.POLICY


def test_dates_duration_and_validation(events_df):
    ev = eh.prepare_events(events_df)
    assert list(ev["duration_months"]) == [2, 2, 1]
    bad = events_df.copy(); bad.loc[0, "end_date"] = "2021-03-01"
    with pytest.raises(DataError):
        eh.prepare_events(bad)


def test_calendar_scopes(events_df):
    ev = eh.prepare_events(events_df)
    idx = pd.date_range("2021-01-01", periods=24, freq="MS")
    all_cal = eh.build_calendar(ev, idx, "ALL")
    d1 = eh.build_calendar(ev, idx, "D01"); d2 = eh.build_calendar(ev, idx, "D02")
    assert all_cal["hiring_freeze"].sum() == 2 and all_cal["hiring_drive"].sum() == 2     # company sees every drive
    assert d1["hiring_drive"].sum() == 2 and d2["hiring_drive"].sum() == 0                 # D02 is not in EV-B
    assert d2["hiring_freeze"].sum() == 2                                                  # ALL-scoped freeze hits every department
    assert all_cal["hiring_freeze"].loc["2021-04-01"] == 1 and all_cal["hiring_freeze"].loc["2021-06-01"] == 0


def test_policy_events_create_no_dummy(events_df):
    ev = eh.prepare_events(events_df)
    cal = eh.build_calendar(ev, pd.date_range("2022-06-01", periods=1, freq="MS"), "ALL")
    assert cal.sum().sum() == 0 or cal["hiring_drive"].sum() == 0


def test_effect_estimates_are_descriptive_ratios(synth, events_df):
    h = pp.clean_history(synth); ev = eh.prepare_events(events_df)
    eff = eh.estimate_event_effects(h, ev)
    assert len(eff) == 3 and (eff["applied_to_forecast"] == False).all()
    assert eff["evidence"].str.contains("not causal").all()
    assert {"hires_multiplier", "exits_multiplier"} <= set(eff.columns)


def test_no_future_events_means_no_adjustment(synth, events_df):
    h = pp.clean_history(synth); ev = eh.prepare_events(events_df)
    months = pp.future_index(h["month"].max(), 6)
    adj = eh.future_adjustments(h, ev, eh.load_future_events("/nonexistent.csv"), months, ["ALL", "D01", "D02"])
    assert adj.empty
    fc = pd.DataFrame({"mean": [10.0] * 6, "lower": [8.0] * 6, "upper": [12.0] * 6}, index=months)
    out = eh.apply_adjustments(fc, adj, "D01")
    assert (out["mean"] == 10.0).all() and (out["event_adjustment"] == 0).all()


def test_assumed_future_event_is_applied_once_and_cumulates(synth, events_df):
    h = pp.clean_history(synth); ev = eh.prepare_events(events_df)
    months = pp.future_index(h["month"].max(), 6)
    fe = pd.DataFrame([{"event_id": "FUT-1", "event_name": "Planned freeze", "start_date": months[1], "end_date": months[2], "department_id": "D01",
                        "hires_multiplier": 0.0, "exits_multiplier": 1.0, "evidence": "assumed"}])
    adj = eh.future_adjustments(h, ev, fe, months, ["ALL", "D01", "D02"])
    d1 = adj[adj.department_id == "D01"].sort_values("month")
    assert len(d1) == 2 and (d1["monthly_effect"] < 0).all()
    assert d1["cumulative_adjustment"].iloc[1] == pytest.approx(d1["monthly_effect"].sum())
    assert adj[adj.department_id == "D02"].empty                                # untouched department
    allr = adj[adj.department_id == "ALL"].sort_values("month")
    assert allr["cumulative_adjustment"].iloc[-1] == pytest.approx(d1["cumulative_adjustment"].iloc[-1])   # company = sum of departments
    fc = pd.DataFrame({"mean": [10.0] * 6, "lower": [8.0] * 6, "upper": [12.0] * 6}, index=months)
    out = eh.apply_adjustments(fc, adj, "D01")
    assert out["mean"].iloc[0] == 10.0 and out["mean"].iloc[-1] < 10.0 and (out[["mean", "lower", "upper"]] >= 0).all().all()
    assert out["mean"].iloc[-1] == pytest.approx(10.0 + d1["cumulative_adjustment"].iloc[-1])   # held after the event ends (not re-added)


def test_historical_events_are_not_reapplied(synth, events_df):
    h = pp.clean_history(synth); ev = eh.prepare_events(events_df)
    months = pp.future_index(h["month"].max(), 6)
    assert eh.future_adjustments(h, ev, pd.DataFrame(columns=eh.FUTURE_COLUMNS), months, ["ALL"]).empty


def test_real_events_all_in_the_past(result):
    assert (result["events"]["end_date"] < result["series"]["ALL"].index[-1] + pd.offsets.MonthEnd(1)).all()
    assert result["adjustments"].empty
