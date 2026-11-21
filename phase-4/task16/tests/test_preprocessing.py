import pandas as pd
import pytest

from cortexhr_task16 import config, preprocessing as pp
from cortexhr_task16.data_loader import DataError


def test_dates_parsed_to_month_start(synth):
    s = synth.copy(); s["month"] = s["month"].dt.strftime("%Y-%m-15")
    out = pp.clean_history(s)
    assert (out["month"].dt.day == 1).all()


def test_exact_duplicates_dropped_conflicts_rejected(synth):
    assert len(pp.clean_history(pd.concat([synth, synth.iloc[:3]]))) == len(synth)
    bad = synth.copy(); extra = synth.iloc[[0]].copy(); extra["headcount_end"] += 5
    with pytest.raises(DataError, match="share"):
        pp.clean_history(pd.concat([bad, extra]))


def test_negative_headcount_rejected(synth):
    s = synth.copy(); s.loc[0, "headcount_end"] = -1
    with pytest.raises(DataError, match="Negative"):
        pp.clean_history(s)


def test_missing_months_interpolated_and_reported(synth):
    h = pp.clean_history(synth)
    h = h[~((h.department_id == "D01") & (h.month == pd.Timestamp("2022-03-01")))]
    s, imputed = pp.get_series(h, "D01")
    assert imputed == ["2022-03"] and not s.isna().any() and len(s) == 60
    assert pp.validate_history(h)["missing_months"] == {"D01": ["2022-03"]}


def test_series_is_monthly_sorted_and_regular(synth):
    s, imputed = pp.get_series(pp.clean_history(synth.sample(frac=1, random_state=1)), "ALL")
    assert s.index.is_monotonic_increasing and s.index.freqstr == "MS" and imputed == []
    assert (s.index.to_series().diff().dropna().dt.days.between(28, 31)).all()


def test_flow_identity_violations_counted(synth):
    s = synth.copy(); s.loc[2, "headcount_end"] += 1
    assert pp.validate_history(pp.clean_history(s))["flow_identity_violations"] >= 1


def test_unknown_department_rejected(synth):
    with pytest.raises(DataError):
        pp.get_series(pp.clean_history(synth), "D99")


def test_real_data_quality(result):
    q = result["quality"]
    assert q["n_months"] == 81 and q["duplicate_keys"] == 0 and q["missing_values"] == 0 and not q["missing_months"]
