"""End-to-end monitoring behaviour: aggregation, failures, missing data, history, privacy, scheduling. Data is SYNTHETIC."""
import csv
import json
from pathlib import Path

import pytest
import yaml

import monitor
from monitoring import drift_monitor, fairness_monitor, monitoring_config as cfg, report_generator as rg, status as st

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT.parents[1] / ".github" / "workflows" / "model-monitoring.yml"
REQUIRED = ("dataset_validation", "drift", "twin_resume", "pay_gap")


def ok_twin(**kw):
    return {"status": "OK", "message": "fine", "valid_pairs": 25, "expected_pairs": 25, "average_gap": 0.0, "maximum_gap": 0.0, **kw}


def ok_pay(**kw):
    comp = {"status": "OK", "message": "fine", "groups": [], "metrics": {}}
    return {"status": "OK", "message": "fine", "components": {"gender": comp, "career_gap": dict(comp)}, "groups_evaluated": 6, **kw}


@pytest.fixture
def fair_ok(monkeypatch):
    monkeypatch.setattr(fairness_monitor, "evaluate_twin_resume", lambda: ok_twin())
    monkeypatch.setattr(fairness_monitor, "evaluate_pay_gap", lambda: ok_pay())


# ---------------------------------------------------------------- aggregation
def F(check, status, item="x"):
    return {"check": check, "item": item, "status": status, "message": "m"}


def test_aggregate_ok():
    a = st.aggregate([F(c, "OK") for c in REQUIRED], REQUIRED)
    assert a["status"] == "OK" and a["reasons"] == [] and a["execution_ok"] and a["evidence_complete"]


def test_aggregate_watch_and_alert_and_categories():
    w = st.aggregate([F("drift", "WATCH"), F("twin_resume", "OK")], REQUIRED)
    assert w["status"] == "WATCH" and w["reasons"][0]["category"] == "drift"
    a = st.aggregate([F("drift", "WATCH"), F("pay_gap", "ALERT")], REQUIRED)
    assert a["status"] == "ALERT" and {r["category"] for r in a["reasons"]} == {"drift", "fairness"}


def test_missing_evidence_is_alert_if_required_watch_if_optional():
    assert st.aggregate([F("twin_resume", "NOT_EVALUATED")], REQUIRED)["status"] == "ALERT"
    a = st.aggregate([F("twin_resume", "NOT_EVALUATED")], ("drift",))
    assert a["status"] == "WATCH" and a["evidence_complete"] is False and a["execution_ok"] is True


def test_execution_error_is_alert_and_distinct_from_drift():
    a = st.aggregate([F("drift", "ERROR")], REQUIRED)
    assert a["status"] == "ALERT" and a["execution_ok"] is False and a["reasons"][0]["category"] == "execution_failure"


def test_unknown_status_fails_closed_and_identical_findings_are_grouped():
    assert st.aggregate([F("drift", "SURPRISE")], REQUIRED)["status"] == "ALERT"
    a = st.aggregate([F("drift", "NOT_EVALUATED", "a"), F("drift", "NOT_EVALUATED", "b")], REQUIRED)
    assert len(a["reasons"]) == 1 and a["reasons"][0]["item"] == "a, b"


def test_worst_and_empty():
    assert st.worst(["OK", "WATCH"]) == "WATCH" and st.worst(["OK", "NOT_EVALUATED"]) == "NOT_EVALUATED" and st.worst([]) == "NOT_EVALUATED"
    assert st.worst(["ALERT", "ERROR"]) == "ERROR"


def test_optional_checks_env(monkeypatch):
    monkeypatch.setenv("CORTEXHR_MONITOR_OPTIONAL_CHECKS", "twin_resume, pay_gap")
    assert cfg.required_checks() == ("dataset_validation", "drift")


# ---------------------------------------------------------------- full runs
def test_overall_ok(tmp_path, drift_csvs, fair_ok):
    drift_csvs()
    rep, code = monitor.run_monitor(out_dir=tmp_path / "o")
    assert rep["status"] == "OK" and code == 0 and rep["drift"]["counts"]["OK"] == 8 and rep["reasons"] == []
    assert rep["drift"]["features_checked"] == 8 and all(f["reference_n"] == 2000 and f["current_n"] == 1000 for f in rep["drift"]["features"])


def test_overall_watch(tmp_path, drift_csvs, fair_ok):
    drift_csvs(cur_kwargs={"remote_share": 0.4})                       # work_mode 20% -> 40% remote: PSI ~0.2
    rep, code = monitor.run_monitor(out_dir=tmp_path / "o")
    wm = next(f for f in rep["drift"]["features"] if f["feature"] == "work_mode")
    assert 0.10 <= wm["psi"] <= 0.25 and wm["status"] == "WATCH"
    assert rep["status"] == "WATCH" and code == 0 and monitor.exit_code_for(rep, "watch") == 1


def test_overall_alert_from_drift(tmp_path, drift_csvs, fair_ok):
    drift_csvs(cur_kwargs={"shift": 0.5, "remote_share": 0.6, "missing_engagement": 0.6})
    rep, code = monitor.run_monitor(out_dir=tmp_path / "o")
    assert rep["status"] == "ALERT" and code == 2 and rep["drift"]["counts"]["ALERT"] >= 3
    assert any(r["category"] == "drift" and r["status"] == "ALERT" for r in rep["reasons"])
    assert next(f for f in rep["drift"]["features"] if f["feature"] == "last_engagement_score")["current_missing_pct"] > 50


def test_fairness_alert_makes_overall_alert(tmp_path, drift_csvs, monkeypatch):
    drift_csvs()
    monkeypatch.setattr(fairness_monitor, "evaluate_twin_resume", lambda: ok_twin(status="ALERT", maximum_gap=0.05))
    monkeypatch.setattr(fairness_monitor, "evaluate_pay_gap", lambda: ok_pay())
    rep, code = monitor.run_monitor(out_dir=tmp_path / "o")
    assert rep["status"] == "ALERT" and code == 2 and rep["reasons"][0]["category"] == "fairness"


def test_no_datasets_is_alert_not_ok(tmp_path):
    rep, code = monitor.run_monitor(out_dir=tmp_path / "o")
    assert rep["status"] == "ALERT" and code == 2 and rep["evidence_complete"] is False and rep["execution_ok"] is True
    assert monitor.exit_code_for(rep, "never") == 2
    assert {r["category"] for r in rep["reasons"]} == {"missing_evidence"}
    assert rep["drift"]["counts"]["NOT_EVALUATED"] == 8 and rep["drift"]["max_psi"] is None


def test_fail_on_never_only_suppresses_complete_alerts():
    complete_alert = {"execution_status": "COMPLETED", "execution_ok": True, "evidence_complete": True, "status": "ALERT"}
    assert monitor.exit_code_for(complete_alert, "never") == 0


def test_one_missing_dataset_and_missing_column(tmp_path, drift_csvs, fair_ok, monkeypatch):
    drift_csvs()
    monkeypatch.setenv("CORTEXHR_SCORING_CSV", str(tmp_path / "nope.csv"))
    rep, _ = monitor.run_monitor(out_dir=tmp_path / "o")
    assert rep["status"] == "ALERT" and rep["validation"]["status"] == "NOT_EVALUATED"
    import pandas as pd
    drift_csvs()
    pd.read_csv(tmp_path / "cur.csv").drop(columns=["work_mode"]).to_csv(tmp_path / "cur.csv", index=False)
    rep, _ = monitor.run_monitor(out_dir=tmp_path / "o2")
    wm = next(f for f in rep["drift"]["features"] if f["feature"] == "work_mode")
    assert rep["status"] == "ALERT" and wm["status"] == "NOT_EVALUATED" and rep["drift"]["counts"]["OK"] == 7


def test_invalid_feature_values_are_an_error_not_ok(tmp_path, drift_csvs, fair_ok):
    import pandas as pd
    drift_csvs()
    d = pd.read_csv(tmp_path / "cur.csv").astype({"tenure_months": object})
    d.loc[0, "tenure_months"] = "unknown"
    d.to_csv(tmp_path / "cur.csv", index=False)
    rep, code = monitor.run_monitor(out_dir=tmp_path / "o")
    assert rep["status"] == "ALERT" and code == 3 and rep["execution_ok"] is False


def test_too_few_rows_is_a_data_integrity_alert(tmp_path, drift_csvs, fair_ok):
    drift_csvs(n_cur=10)
    rep, _ = monitor.run_monitor(out_dir=tmp_path / "o")
    assert rep["status"] == "ALERT" and rep["validation"]["status"] == "ALERT"


def test_failed_check_never_reported_as_ok(tmp_path, drift_csvs, fair_ok, monkeypatch):
    drift_csvs()
    monkeypatch.setattr(drift_monitor, "run_drift", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom with secret /tmp/x")))
    rep, code = monitor.run_monitor(out_dir=tmp_path / "o")
    assert rep["status"] == "ALERT" and code == 3 and "secret" not in json.dumps(rep) and rep["drift"]["status"] == "ERROR"


def test_crash_publishes_failed_report_and_replaces_stale_ok(tmp_path, drift_csvs, fair_ok, monkeypatch):
    drift_csvs()
    out = tmp_path / "o"
    first, _ = monitor.run_monitor(out_dir=out)
    assert first["status"] == "OK"
    monkeypatch.setattr(rg, "build_report", lambda *a, **k: (_ for _ in ()).throw(MemoryError()))
    rep, code = monitor.run_monitor(out_dir=out)
    latest = json.loads((out / "latest_report.json").read_text())
    assert rep["execution_status"] == "FAILED" and code == 3 and latest["status"] == "ALERT" and latest["execution_status"] == "FAILED"
    assert latest["run_id"] != first["run_id"]


def test_persistence_failure_is_visible(tmp_path, drift_csvs, fair_ok, monkeypatch):
    drift_csvs()
    monkeypatch.setattr(rg, "publish", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    rep, code = monitor.run_monitor(out_dir=tmp_path / "o")
    assert rep["status"] == "ALERT" and code == 3 and any(r["check"] == "persistence" for r in rep["reasons"])


def test_check_subset_marks_the_rest_not_evaluated(tmp_path, drift_csvs):
    drift_csvs()
    rep, _ = monitor.run_monitor(("drift",), tmp_path / "o")
    assert rep["fairness"]["twin_resume"]["status"] == "NOT_EVALUATED" and rep["status"] == "ALERT"


def test_cli(tmp_path, capsys, drift_csvs, fair_ok):
    drift_csvs()
    assert monitor.main(["--output-dir", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for needle in ("CORTEXHR - MODEL MONITORING REPORT", "DRIFT MONITORING", "FAIRNESS MONITORING", "OVERALL STATUS", "STATUS: OK", "Pairs Evaluated: 25 of 25",
                   "Feature", "tenure_months"):
        assert needle in out
    with pytest.raises(SystemExit):
        monitor.main(["--checks", "bogus"])
    assert monitor.main(["--output-dir", str(tmp_path / "o2"), "--checks", "drift", "--fail-on", "never"]) == 2     # unselected required checks remain missing evidence


# ---------------------------------------------------------------- outputs and history
def test_outputs_exist_valid_and_stable_schema(tmp_path, drift_csvs, fair_ok):
    drift_csvs()
    out = tmp_path / "o"
    rep, _ = monitor.run_monitor(out_dir=out)
    for name in ("latest_report.json", "latest_report.md", "psi_results.csv", "fairness_results.json", "monitoring_history.csv"):
        assert (out / name).is_file(), name
    assert json.loads((out / "latest_report.json").read_text())["run_id"] == rep["run_id"]
    json.loads((out / "fairness_results.json").read_text())
    rows = list(csv.reader((out / "psi_results.csv").open()))
    assert rows[0] == rg.PSI_COLUMNS and len(rows) == 9 and all(len(r) == len(rg.PSI_COLUMNS) for r in rows)
    assert next(csv.reader((out / "monitoring_history.csv").open())) == rg.HISTORY_COLUMNS
    assert "# CortexHR model monitoring report" in (out / "latest_report.md").read_text()


def test_history_is_preserved_and_never_overwritten(tmp_path, drift_csvs, fair_ok):
    drift_csvs()
    out = tmp_path / "o"
    r1, _ = monitor.run_monitor(out_dir=out)
    first_text = (out / "history" / r1["run_id"] / "report.json").read_text()
    drift_csvs(cur_kwargs={"remote_share": 0.6})
    r2, _ = monitor.run_monitor(out_dir=out)
    assert r1["run_id"] != r2["run_id"] and len(list((out / "history").iterdir())) == 2
    assert (out / "history" / r1["run_id"] / "report.json").read_text() == first_text
    rows = list(csv.DictReader((out / "monitoring_history.csv").open()))
    assert [r["run_id"] for r in rows] == [r1["run_id"], r2["run_id"]] and [r["status"] for r in rows] == ["OK", "ALERT"]
    with pytest.raises(FileExistsError):
        rg.publish(r1, out)                                            # an existing run id is never overwritten


def test_history_with_old_schema_is_kept_not_overwritten(tmp_path, drift_csvs, fair_ok):
    drift_csvs()
    out = tmp_path / "o"
    out.mkdir()
    (out / "monitoring_history.csv").write_text("run_id,status\nold,OK\n")
    r, _ = monitor.run_monitor(out_dir=out)
    legacy = list(out.glob("monitoring_history.legacy-*.csv"))
    assert len(legacy) == 1 and "old,OK" in legacy[0].read_text()
    assert len(list(csv.DictReader((out / "monitoring_history.csv").open()))) == 1


def test_no_partial_files_left_behind(tmp_path, drift_csvs, fair_ok):
    drift_csvs()
    monitor.run_monitor(out_dir=tmp_path / "o")
    assert not list((tmp_path / "o").rglob("*.tmp"))


def test_report_contains_no_individual_data_or_paths(tmp_path, drift_csvs, fair_ok):
    drift_csvs()
    out = tmp_path / "o"
    monitor.run_monitor(out_dir=out)
    blob = "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    assert "X-1-" not in blob and "X-2-" not in blob and "employee_id" not in blob and str(tmp_path) not in blob


def test_json_is_strictly_valid_even_with_nan(tmp_path):
    r = rg.build_failure_report("rid", "2026-01-01T00:00:00Z", "X")
    r["drift"]["max_psi"] = float("nan")
    rg.publish(r, tmp_path)
    assert json.loads((tmp_path / "latest_report.json").read_text(), parse_constant=lambda c: pytest.fail("NaN in JSON"))["drift"]["max_psi"] is None


# ---------------------------------------------------------------- scheduling
@pytest.fixture(scope="module")
def wf():
    return yaml.safe_load(WORKFLOW.read_text())


def test_workflow_yaml_is_valid_and_scheduled(wf):
    on = wf.get("on", wf.get(True))                                    # YAML 1.1 parses the key `on` as True
    assert wf["name"] == "model-monitoring" and on["schedule"] == [{"cron": "0 2 * * *"}] and "workflow_dispatch" in on


def test_workflow_runs_monitor_with_python_and_dependencies(wf):
    steps = wf["jobs"]["monitor"]["steps"]
    runs = "\n".join(s.get("run", "") for s in steps)
    assert "python monitor.py" in runs and "pip install -r requirements.txt" in runs
    assert wf["jobs"]["monitor"]["defaults"]["run"]["working-directory"] == "phase-4/task18"
    assert any(s.get("uses", "").startswith("actions/setup-python") for s in steps) and (ROOT / "monitor.py").is_file() and (ROOT / "requirements.txt").is_file()


def test_workflow_uploads_artifacts_always_and_fails_visibly(wf):
    steps = wf["jobs"]["monitor"]["steps"]
    up = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact"))
    assert up["if"] == "always()" and up["with"]["path"] == "phase-4/task18/outputs/monitoring/" and up["with"]["if-no-files-found"] == "error"
    run = next(s for s in steps if "python monitor.py" in s.get("run", ""))
    assert not run.get("continue-on-error") and "if" not in run          # a non-zero exit fails the job


def test_workflow_does_not_leak_secrets_or_write(wf):
    text = WORKFLOW.read_text()
    assert wf["permissions"]["contents"] == "read" and "echo ${{ secrets" not in text and "set -x" not in text
    fetch = next(s for s in wf["jobs"]["monitor"]["steps"] if s.get("name", "").startswith("Fetch private monitoring data"))
    assert fetch["continue-on-error"] is True and "download_monitoring_data.py" in fetch["run"]
    assert fetch["env"] == {
        "DATA_URL": "${{ secrets.CORTEXHR_MONITORING_DATA_URL }}",
        "DATA_SHA256": "${{ secrets.CORTEXHR_MONITORING_DATA_SHA256 }}",
    }
    assert "::add-mask::" in (ROOT / "download_monitoring_data.py").read_text()
    ignore = (ROOT.parents[1] / ".gitignore").read_text()
    assert "phase-4/task18/outputs/monitoring/" in ignore and "phase-4/task18/data/" in ignore        # results / private data are not committed


# ---------------------------------------------------------------- population-matched reference policy
def _mixed_reference(tmp_path, monkeypatch, ext_kwargs=None, n_matched=400, cur_kwargs=None):
    """Reference = 2000 'historical_extension' rows (different population) + n_matched 'master_employee_snapshot' rows (same population as scoring)."""
    import pandas as pd
    import synthetic
    ref = pd.concat([synthetic.attrition_frame(2000, seed=1, source="historical_extension", **(ext_kwargs or {"shift": 0.5})),
                     synthetic.attrition_frame(n_matched, seed=3, source="master_employee_snapshot")], ignore_index=True)
    ref.to_csv(tmp_path / "ref.csv", index=False)
    synthetic.attrition_frame(430, seed=2, **(cur_kwargs or {})).to_csv(tmp_path / "cur.csv", index=False)
    monkeypatch.setenv("CORTEXHR_REFERENCE_CSV", str(tmp_path / "ref.csv"))
    monkeypatch.setenv("CORTEXHR_SCORING_CSV", str(tmp_path / "cur.csv"))


def test_matched_reference_drives_the_verdict_and_training_skew_is_held_at_watch_not_hidden(tmp_path, monkeypatch, fair_ok):
    _mixed_reference(tmp_path, monkeypatch)
    rep, _ = monitor.run_monitor(out_dir=tmp_path / "o")
    t = next(f for f in rep["drift"]["features"] if f["feature"] == "tenure_months")
    assert t["reference_used"] == "matched" and t["reference_n"] == 400 and t["training_reference_n"] == 2400
    assert t["status_training_reference"] == "ALERT" and t["psi_training_reference"] > 0.25      # the training-serving skew stays visible
    assert t["psi"] < 0.10 and t["reference_internal_psi"] >= 0.10                                  # sub-populations differ -> composition
    assert t["status"] == "WATCH" and "training-serving skew" in t["message"]                      # held at WATCH: not OK, not hidden
    assert rep["status"] == "WATCH" and rep["drift"]["reference_policy"]["primary"] == "matched"
    assert any(r["category"] == "drift" for r in rep["reasons"])


def test_genuine_shift_against_the_matched_reference_is_still_an_alert(tmp_path, monkeypatch, fair_ok):
    _mixed_reference(tmp_path, monkeypatch, ext_kwargs={}, cur_kwargs={"remote_share": 0.6, "shift": 0.5})
    rep, code = monitor.run_monitor(out_dir=tmp_path / "o")
    wm = next(f for f in rep["drift"]["features"] if f["feature"] == "work_mode")
    assert wm["status"] == "ALERT" and wm["reference_used"] == "matched" and rep["status"] == "ALERT" and code == 2


def test_matched_reference_too_small_falls_back_to_the_full_reference_visibly(tmp_path, monkeypatch, fair_ok):
    _mixed_reference(tmp_path, monkeypatch, n_matched=50)
    rep, _ = monitor.run_monitor(out_dir=tmp_path / "o")
    assert rep["drift"]["reference_policy"]["primary"] == "full"
    assert all(f["reference_used"] == "full" and f["reference_n"] == 2050 for f in rep["drift"]["features"])


@pytest.mark.parametrize("primary,training,internal,final", [
    ("OK", "ALERT", 0.50, "WATCH"),        # explained by composition: held at WATCH
    ("OK", "ALERT", 0.02, "ALERT"),        # NOT explained: the stricter status stands
    ("OK", "ALERT", None, "ALERT"),        # no evidence of composition: stricter status stands
    ("OK", "WATCH", 0.50, "OK"),           # explained, mild: note only
    ("OK", "WATCH", 0.02, "WATCH"),
    ("WATCH", "ALERT", 0.50, "WATCH"),
    ("ALERT", "WATCH", 0.50, "ALERT"),     # the matched reference is never made weaker by the training one
    ("ALERT", "OK", None, "ALERT"),
    ("OK", None, None, "OK")])
def test_reconcile_rules(primary, training, internal, final):
    assert drift_monitor._reconcile(primary, training, internal)[0] == final


def test_psi_results_csv_has_the_reference_columns(tmp_path, monkeypatch, fair_ok):
    _mixed_reference(tmp_path, monkeypatch)
    monitor.run_monitor(out_dir=tmp_path / "o")
    head = next(csv.reader((tmp_path / "o" / "psi_results.csv").open()))
    assert head == rg.PSI_COLUMNS and {"reference_used", "psi_training_reference", "reference_internal_psi"} <= set(head)
    assert "PSI vs training" in rg.render_console(json.loads((tmp_path / "o" / "latest_report.json").read_text()))


def test_workflow_publishes_to_the_api_and_verifies_without_leaking(wf):
    steps = wf["jobs"]["monitor"]["steps"]
    pub = next(s for s in steps if s.get("name", "").startswith("Publish to the deployed API"))
    assert pub["if"] == "always()" and "python verify_live.py --publish" in pub["run"]
    assert {"CORTEXHR_API_URL", "CORTEXHR_API_KEY", "CORTEXHR_MONITORING_PUBLISH_KEY"} <= set(pub["env"]) and all("secrets." in v for v in pub["env"].values())
    assert "echo $CORTEXHR_MONITORING_PUBLISH_KEY" not in pub["run"] and (ROOT / "verify_live.py").is_file()
    assert "::add-mask::" in (ROOT / "verify_live.py").read_text()
    order = [s.get("name", s.get("uses", "")) for s in steps]
    assert order.index("Run monitoring") < order.index(pub["name"]) < order.index("Upload monitoring results")


def test_verify_live_compare_detects_mismatch_and_overexposure(monkeypatch):
    import verify_live
    monkeypatch.delenv("CORTEXHR_API_URL", raising=False)
    local = {"status": "WATCH", "timestamp_utc": "2026-01-01T00:00:00Z", "fairness": {"twin_resume": {"status": "OK"}, "pay_gap": {"status": "OK"}},
             "drift": {"counts": {"ALERT": 0, "WATCH": 2}}}
    live = {"status": "WATCH", "timestamp": "2026-01-01T00:00:00Z", "fairness": {"twin_resume": {"status": "OK"}, "pay_gap": {"status": "OK"}},
            "drift": {"alert": 0, "watch": 2}}
    assert verify_live.compare(local, live) == []
    assert verify_live.compare(local, {**live, "status": "OK"}) and verify_live.compare(local, {**live, "timestamp": "x"})
    assert verify_live.compare(local, {**live, "salary": 1}) == ["unexpected fields exposed: ['salary']"]
    assert verify_live.compare(local, {**live, "fairness": {"twin_resume": {"status": "ALERT"}, "pay_gap": {"status": "OK"}}})
    assert verify_live.main([]) == 2                                      # no URL/keys configured -> misconfiguration, never "verified"


@pytest.mark.parametrize("url", ("http://api.example.test", "https://api.example.test:invalid"))
def test_verify_live_rejects_invalid_api_url(monkeypatch, url):
    import verify_live
    monkeypatch.setenv("CORTEXHR_API_URL", url)
    monkeypatch.setenv("CORTEXHR_API_KEY", "test-key")
    assert verify_live.main([]) == 2


def test_verify_live_redirect_handler_rejects_insecure_or_cross_origin_redirects():
    import urllib.request
    import verify_live
    handler = verify_live.SameOriginHTTPSRedirectHandler()
    request = urllib.request.Request("https://api.example.test/v1/health")
    assert handler.redirect_request(request, None, 302, "Found", {}, "http://api.example.test/v1/health") is None
    assert handler.redirect_request(request, None, 302, "Found", {}, "https://other.example.test/v1/health") is None
