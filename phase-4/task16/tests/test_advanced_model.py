import numpy as np
import pandas as pd
import pytest

from cortexhr_task16 import advanced_forecaster as af, preprocessing as pp


@pytest.fixture
def series_cal(synth):
    from cortexhr_task16.event_handler import EFFECT_DUMMIES
    s, _ = pp.get_series(pp.clean_history(synth), "ALL")
    return s, pd.DataFrame(0.0, index=s.index, columns=EFFECT_DUMMIES)


@pytest.mark.parametrize("spec", af.CANDIDATES, ids=lambda s: s.name)
def test_every_candidate_gives_valid_six_month_forecast(spec, series_cal):
    y, cal = series_cal
    fc = af.run_candidate(spec, y, cal, 6)
    assert len(fc) == 6 and fc.index[0] == y.index[-1] + pd.offsets.MonthBegin(1)
    assert (fc["lower"] <= fc["mean"]).all() and (fc["mean"] <= fc["upper"]).all()
    assert (fc >= 0).all().all() and not fc.isna().any().any()


def test_interval_widens_with_horizon(series_cal):
    y, cal = series_cal
    fc = af.run_candidate(af.SPECS["drift_events"], y, cal, 6)
    w = (fc["upper"] - fc["lower"]).values
    assert (np.diff(w) > 0).all()


def test_linear_series_is_extrapolated(trend_series):
    from cortexhr_task16.event_handler import EFFECT_DUMMIES
    cal = pd.DataFrame(0.0, index=trend_series.index, columns=EFFECT_DUMMIES)
    fc = af.run_candidate(af.SPECS["drift_events"], trend_series, cal, 6)
    assert np.allclose(fc["mean"].values, trend_series.iloc[-1] + 2.0 * np.arange(1, 7))


def test_event_dummy_is_estimated_not_ignored():
    idx = pd.date_range("2020-01-01", periods=60, freq="MS")
    rng = np.random.default_rng(1)
    change = 3 + rng.normal(0, 0.3, 60); freeze = np.zeros(60); freeze[20:23] = 1; change -= 4 * freeze
    y = pd.Series(100 + np.cumsum(change), index=idx)
    cal = pd.DataFrame({"hiring_freeze": freeze, "hiring_drive": 0.0}, index=idx)
    with_dummy = af.run_candidate(af.SPECS["drift_events"], y, cal, 6)
    without = af.run_candidate(af.SPECS["drift_events"], y, cal.assign(hiring_freeze=0.0), 6)
    # without the dummy the freeze drags the estimated drift down; with it the drift is ~+3/month
    assert abs(with_dummy["mean"].iloc[-1] - (y.iloc[-1] + 18)) < abs(without["mean"].iloc[-1] - (y.iloc[-1] + 18))


def test_selection_prefers_simplest_within_tolerance():
    cv = pd.DataFrame({"model": ["a", "b", "c", "seasonal_baseline"], "complexity": [3, 1, 2, 0], "cv_mae": [5.0, 5.1, 9.0, 1.0]})
    assert af.select_model(cv, tolerance=0.03) == "b"          # b is 2% worse than a but simpler; baseline never eligible
    assert af.select_model(cv, tolerance=0.0) == "a"


def test_cv_uses_only_training_window(series_cal):
    y, cal = series_cal
    cv1 = af.cross_validate(y.iloc[:48], cal.iloc[:48], 6, specs=[af.SPECS["drift_events"]])
    y2 = y.copy(); y2.iloc[48:] = 9999
    cv2 = af.cross_validate(y2.iloc[:48], cal.iloc[:48], 6, specs=[af.SPECS["drift_events"]])
    assert cv1["cv_mae"].iloc[0] == cv2["cv_mae"].iloc[0]


def test_continuity_check_flags_wild_jumps(series_cal):
    y, _ = series_cal
    ok = pd.DataFrame({"mean": y.iloc[-1] + np.arange(1, 7)})
    bad = pd.DataFrame({"mean": y.iloc[-1] + 500 + np.arange(1, 7)})
    assert af.continuity_check(y, ok)["ok"] and not af.continuity_check(y, bad)["ok"]
