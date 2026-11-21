import numpy as np
import pandas as pd

from cortexhr_task16 import baseline_forecaster as bf


def test_formula_same_month_last_year_plus_trend(trend_series):
    y = trend_series
    f = bf.point_forecast(y, 6)
    # linear +2/month => annual trend 24; forecast(t) = y(t-12) + 24 = continues the line
    assert np.allclose(f.values, y.values[-1] + 2.0 * np.arange(1, 7))
    assert bf.annual_trend(y) == 24.0


def test_horizon_and_dates(trend_series):
    f = bf.point_forecast(trend_series, 6)
    assert len(f) == 6 and f.index[0] == trend_series.index[-1] + pd.offsets.MonthBegin(1)
    assert (f.index.to_series().diff().dropna().dt.days.between(28, 31)).all()


def test_no_future_leakage(trend_series):
    y = trend_series.copy()
    a = bf.forecast(y.iloc[:40], 6)
    y2 = y.copy(); y2.iloc[40:] = 10_000                    # wreck everything after the origin
    b = bf.forecast(y2.iloc[:40], 6)
    pd.testing.assert_frame_equal(a, b)


def test_short_series_falls_back_without_error():
    y = pd.Series([10, 11, 12, 13, 14.0], index=pd.date_range("2024-01-01", periods=5, freq="MS"))
    f = bf.forecast(y, 6)
    assert len(f) == 6 and (f["mean"] > y.iloc[-1]).all() and (f["lower"] <= f["mean"]).all() and (f["mean"] <= f["upper"]).all()


def test_nonnegative_even_for_shrinking_series():
    y = pd.Series(np.maximum(60 - 3 * np.arange(40), 0).astype(float), index=pd.date_range("2020-01-01", periods=40, freq="MS"))
    f = bf.forecast(y, 6)
    assert (f >= 0).all().all()


def test_intervals_ordered_on_noisy_series(synth):
    from cortexhr_task16 import preprocessing as pp
    s, _ = pp.get_series(pp.clean_history(synth), "ALL")
    f = bf.forecast(s, 6)
    assert (f["lower"] <= f["mean"]).all() and (f["mean"] <= f["upper"]).all()
