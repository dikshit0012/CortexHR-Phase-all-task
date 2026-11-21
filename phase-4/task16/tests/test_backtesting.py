import numpy as np
import pandas as pd
import pytest

from cortexhr_task16 import backtesting as bt, preprocessing as pp
from cortexhr_task16.event_handler import EFFECT_DUMMIES


def test_metrics_match_hand_calculation():
    a, f = [100, 200, 300], [110, 190, 330]
    m = bt.metrics(a, f, [90, 180, 280], [120, 195, 320])
    assert m["MAE"] == pytest.approx(50 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert m["WAPE_pct"] == pytest.approx(50 / 600 * 100)
    assert m["MAPE_pct"] == pytest.approx(np.mean([0.1, 0.05, 0.1]) * 100)
    assert m["bias"] == pytest.approx(10.0)                         # (+10 -10 +30)/3, positive = over-forecast
    assert m["inside_80"] == 2 and m["coverage_80_pct"] == pytest.approx(200 / 3)   # 100 and 300 inside, 200 outside [180,195]
    assert m["mean_width_80"] == pytest.approx((30 + 15 + 40) / 3)


def test_split_is_chronological_and_six_months(trend_series):
    tr, te = bt.split_holdout(trend_series.sample(frac=1, random_state=0), 6)
    assert len(te) == 6 and tr.index.max() < te.index.min() and len(tr) == 42


def test_too_short_series_rejected():
    y = pd.Series(range(20), index=pd.date_range("2024-01-01", periods=20, freq="MS"), dtype=float)
    with pytest.raises(ValueError):
        bt.split_holdout(y, 6)


def test_backtest_is_leakage_free(synth):
    s, _ = pp.get_series(pp.clean_history(pp.pd.concat([synth, synth.assign(month=synth.month + pd.DateOffset(years=5))]).drop_duplicates(["month", "department_id"])), "D01")
    cal = pd.DataFrame(0.0, index=s.index, columns=EFFECT_DUMMIES)
    a = bt.backtest_series(s, cal, "drift_events")
    s2 = s.copy(); s2.iloc[-6:] = s2.iloc[-6:] * 5 + 1000            # corrupt only the hidden months
    b = bt.backtest_series(s2, cal, "drift_events")
    cols = [c for c in a["results"].columns if "forecast" in c or "lower" in c or "upper" in c]
    pd.testing.assert_frame_equal(a["results"][cols], b["results"][cols])
    assert (a["results"]["actual"] != b["results"]["actual"]).all()
    assert a["train_end"] == s.index[-7]


def test_backtest_model_selection_ignores_holdout(result):
    # selection on the real data must equal selection computed on the training months alone
    from cortexhr_task16 import advanced_forecaster as af
    y = result["series"]["ALL"]; cal = result["calendars"]["ALL"]
    tr = y.iloc[:-6]
    assert af.select_model(af.cross_validate(tr, cal.loc[tr.index])) == result["backtest_model"]


def test_real_backtest_table_consistency(result):
    b = result["backtest"]["ALL"]["results"]
    y = result["series"]["ALL"]
    assert list(b["month"]) == list(y.index[-6:]) and list(b["actual"]) == list(y.iloc[-6:])
    assert np.allclose(b["advanced_abs_error"], (b["advanced_forecast"] - b["actual"]).abs())
    assert (b["advanced_lower_80"] <= b["advanced_forecast"]).all() and (b["advanced_forecast"] <= b["advanced_upper_80"]).all()
