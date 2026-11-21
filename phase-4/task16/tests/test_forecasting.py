import numpy as np
import pandas as pd

from cortexhr_task16 import config, forecasting as fc, backtesting as bt


def test_contract_matches_sample_exactly(result):
    out, sample = result["contract"], result["sample"]
    assert fc.validate_contract(out, sample) == []
    assert list(out.columns) == list(sample.columns) and len(out) == len(sample) == 72
    assert out["forecast_id"].tolist() == sample["forecast_id"].tolist()
    assert out[["metric", "department_id", "forecast_month"]].equals(sample[["metric", "department_id", "forecast_month"]])
    for c in ("forecast_value", "lower_80", "upper_80"):
        assert out[c].dtype == sample[c].dtype


def test_horizon_is_six_consecutive_months(result):
    for _, g in result["contract"].groupby(["metric", "department_id"]):
        m = pd.to_datetime(g["forecast_month"])
        assert len(g) == 6 and (m.dt.day == 1).all()
        assert list(m) == list(pd.date_range("2026-10-01", periods=6, freq="MS"))
    assert (result["contract"]["trained_through"] == "2026-09").all()


def test_bounds_and_non_negative(result):
    o = result["contract"]
    assert (o["lower_80"] <= o["forecast_value"]).all() and (o["forecast_value"] <= o["upper_80"]).all()
    assert (o[["forecast_value", "lower_80", "upper_80"]] >= 0).all().all()
    hc = o[o["metric"] == "headcount_end"]
    assert (hc["upper_80"] > hc["lower_80"]).all()


def test_forecast_continuity_with_history(result):
    assert result["continuity"]["ok"]
    first = result["headcount_fc"]["ALL"]["mean"].iloc[0]
    assert abs(first - result["series"]["ALL"].iloc[-1]) <= result["continuity"]["max_historical_step"]


def test_final_model_trained_on_all_history_and_no_event_adjustment(result):
    assert result["series"]["ALL"].index[-1] == pd.Timestamp("2026-09-01")
    assert (result["headcount_fc"]["ALL"]["event_adjustment"] == 0).all()


def test_exits_forecast_consistent(result):
    e = result["exits_fc"]
    assert len(e) == 6 and (e["lower_80"] <= e["expected_exits"]).all() and (e["expected_exits"] <= e["upper_80"]).all()
    assert np.allclose(e["implied_hires_needed"], e["expected_net_change"] + e["expected_exits"])


def test_backtest_metrics_recomputed_independently(result):
    results, mets = fc.backtest_tables(result)
    r = results[results.department_id == "ALL"]
    row = mets[(mets.department_id == "ALL") & (mets.model == "Advanced Model")].iloc[0]
    assert row["MAE"] == np.mean(np.abs(r["advanced_forecast"] - r["actual"]))
    assert row["n"] == 6


def test_reliability_assessment_is_data_driven(result):
    rel = result["reliability"]
    t = rel["table"]
    assert rel["widest"]["month"] == t.loc[t["width_80"].idxmax(), "month"]
    assert len(t) == 6


def test_outputs_and_charts_are_written(result, tmp_path, monkeypatch):
    for name in ("OUTPUTS", "CHARTS", "REPORTS", "ROOT"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    (tmp_path / "root").mkdir()
    fc.write_outputs(result)
    for f in ("headcount_forecast.csv", "backtest_results.csv", "backtest_metrics.csv", "workforce_event_adjustments.csv", "expected_exits_forecast.csv"):
        assert (tmp_path / "outputs" / f).stat().st_size > 0
    written = pd.read_csv(tmp_path / "outputs" / "headcount_forecast.csv")
    assert list(written.columns) == config.SAMPLE_COLUMNS and len(written) == 72
    for f in ("historical_headcount.png", "historical_exits.png", "backtest_comparison.png", "six_month_forecast.png", "model_performance.png"):
        assert (tmp_path / "charts" / f).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    ev = pd.read_csv(tmp_path / "outputs" / "workforce_event_adjustments.csv")
    assert (ev["event_id"] == "NONE").sum() == 1
    for f in ("forecast_analysis.md", "forecast_summary.md"):
        assert (tmp_path / "reports" / f).read_text(encoding="utf-8").startswith("# Task 16")
    assert (tmp_path / "root" / "README.md").exists()


def test_cli_reports_missing_workbook(tmp_path, capsys):
    assert fc.main(["--workbook", str(tmp_path / "missing.xlsx"), "--no-write"]) == 2
