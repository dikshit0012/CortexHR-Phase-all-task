"""Task 18 monitoring end to end: REAL monitor.py reports (generated from synthetic data) are published to the REAL patched API with the
write-only key, then read back by the console's client and pages. Report timestamps are rewritten to 'now' so staleness is deterministic."""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

from api_client import ApiClient, ApiError
from config import Config

pytestmark = pytest.mark.live
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "monitor_reports"
APP = str(Path(__file__).resolve().parents[1] / "app.py")
LABEL = "AI-generated, for human review"
WHITELIST = {"status", "timestamp", "execution_status", "execution_ok", "evidence_complete", "drift", "fairness", "reasons", "last_run",
             "report_age_hours", "stale", "request_id"}                         # same whitelist as Task 18's verify_live.py
SCENARIOS = ["ok", "watch", "alert_drift", "alert_fairness", "alert_missing", "stale_ok"]


def report(name, age_hours=0.0):
    r = json.loads((FIX / f"{name}.json").read_text())
    r["timestamp_utc"] = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return r


def publish(m, name, age_hours=None, key=None):
    age = (100.0 if name == "stale_ok" else 0.0) if age_hours is None else age_hours
    r = requests.put(m["base"] + "/v1/monitoring/report", data=json.dumps(report(name, age)),
                     headers={"X-Monitoring-Publish-Key": key or m["pub"], "Content-Type": "application/json"}, timeout=30)
    assert r.status_code == 200, r.text
    return r.json()


def client(m, key=None):
    return ApiClient(Config(api_url=m["base"], api_key=m["key"] if key is None else key, timeout=30))


def err(fn):
    with pytest.raises(ApiError) as e:
        fn()
    return e.value


def txt(at):
    return " ".join(x.value for g in (at.markdown, at.error, at.info, at.warning, at.caption, at.success, at.subheader) for x in g)


@pytest.fixture
def ui(mon, monkeypatch):
    monkeypatch.setenv("CORTEXHR_API_URL", mon["base"]); monkeypatch.setenv("CORTEXHR_API_KEY", mon["key"])
    monkeypatch.setenv("CORTEXHR_TIMEOUT_SECONDS", "60")

    def run(page):
        at = AppTest.from_file(APP, default_timeout=90); at.session_state["page"] = page
        return at.run()
    return run


# ---- nothing published yet --------------------------------------------------------------------------------------------------------------
def test_no_report_yet_is_a_clean_404_not_an_invented_status(mon, ui):
    e = err(client(mon).monitoring)
    assert e.kind == "not_found" and e.code == "monitoring_report_not_found"
    t = txt(ui("Monitoring"))
    assert "No monitoring report has been published" in t and "no status is shown" in t
    assert "Not available yet" in txt(ui("Overview"))


# ---- every real scenario, via the client ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", SCENARIOS)
def test_client_reads_what_monitor_py_produced(mon, name):
    src = report(name, 100.0 if name == "stale_ok" else 0.0)
    publish(mon, name)
    got = client(mon).monitoring()
    c = src["drift"]["counts"]
    assert got["status"] == src["status"] and got["execution_ok"] == src["execution_ok"] and got["evidence_complete"] == src["evidence_complete"]
    assert got["drift"] == {"features_checked": 8, "ok": c["OK"], "watch": c["WATCH"], "alert": c["ALERT"], "not_evaluated": c["NOT_EVALUATED"], "error": c["ERROR"]}
    assert {k: v["status"] for k, v in got["fairness"].items()} == {k: v["status"] for k, v in src["fairness"].items()}
    assert [(r["category"], r["item"], r["status"]) for r in got["reasons"]] == [(r["category"], r["item"][:60], r["status"]) for r in src["reasons"]]
    assert got["stale"] is (name == "stale_ok") and set(got) <= WHITELIST


def test_api_exposes_only_aggregates_never_raw_report_content(mon):
    publish(mon, "alert_drift")
    raw = requests.get(mon["base"] + "/v1/monitoring", headers={"X-API-Key": mon["key"]}, timeout=30).json()
    assert set(raw) <= WHITELIST
    stored = json.loads(mon["file"].read_text())
    for forbidden in ("datasets", "validation", "features", "code_version", "required_checks"):
        assert forbidden not in stored and forbidden not in raw
    assert all(set(r) == {"category", "check", "item", "status"} for r in raw["reasons"])


# ---- the console pages, against the live API ---------------------------------------------------------------------------------------------
def test_monitoring_page_ok(mon, ui):
    publish(mon, "ok"); at = ui("Monitoring"); t = txt(at)
    assert not at.exception and not at.error and "No feature shows drift" in t and LABEL in t and "outdated" not in t
    assert "ALERT" not in " ".join(m.value for m in at.markdown if "badge" in m.value)


def test_monitoring_page_watch_names_work_mode(mon, ui):
    publish(mon, "watch"); at = ui("Monitoring")
    flat = " ".join(" ".join(map(str, d.value.values.flatten())) for d in at.dataframe)
    assert "work_mode" in flat and "WATCH" in " ".join(m.value for m in at.markdown if "badge" in m.value) and not at.exception


def test_monitoring_page_drift_alert_lists_five_features(mon, ui):
    publish(mon, "alert_drift"); at = ui("Monitoring")
    flat = " ".join(" ".join(map(str, d.value.values.flatten())) for d in at.dataframe)
    expect = [r["item"] for r in report("alert_drift")["reasons"] if r["category"] == "drift"]
    assert len(expect) == 5 and all(f in flat for f in expect) and "Escalate" in txt(at)


def test_monitoring_page_fairness_alert_with_clean_drift(mon, ui):
    publish(mon, "alert_fairness"); at = ui("Monitoring")
    badges = [m.value for m in at.markdown if "Twin-resume" in m.value]
    assert badges and "ALERT" in badges[0] and "No feature shows drift" in txt(at)


def test_monitoring_page_missing_evidence_not_shown_as_healthy(mon, ui):
    publish(mon, "alert_missing"); at = ui("Monitoring"); t = txt(at)
    assert "could not be evaluated" in t and "not a pass" in t and "No feature shows drift" not in t
    assert sum("NOT EVALUATED" in m.value for m in at.markdown) >= 2          # both fairness checks
    flat = " ".join(" ".join(map(str, d.value.values.flatten())) for d in at.dataframe)
    assert "Missing evidence" in flat


def test_monitoring_page_stale_report_flagged(mon, ui):
    publish(mon, "stale_ok"); t = txt(ui("Monitoring"))
    assert "outdated" in t and "older than the allowed age" in t


@pytest.mark.parametrize("name,expect", [("watch", "WATCH"), ("alert_drift", "ALERT"), ("ok", ">OK<"), ("stale_ok", "OK · outdated")])
def test_overview_card_shows_live_status(mon, ui, name, expect):
    publish(mon, name); at = ui("Overview")
    cards = " ".join(m.value for m in at.markdown if "Model monitoring" in m.value)
    assert expect in cards and "Not available yet" not in cards and not at.exception


def test_api_key_and_publish_key_never_rendered(mon, ui):
    publish(mon, "alert_missing")
    for page in ("Overview", "Monitoring"):
        shown = repr(ui(page))
        assert mon["key"] not in shown and mon["pub"] not in shown


# ---- write path security (the console never uses it, but the API it depends on must hold) -------------------------------------------------
def test_publish_requires_the_write_only_key(mon):
    body = json.dumps(report("ok"))
    for hdr in ({}, {"X-Monitoring-Publish-Key": "wrong"}, {"X-Monitoring-Publish-Key": mon["key"]}):       # the READ key must not publish
        assert requests.put(mon["base"] + "/v1/monitoring/report", data=body, headers=hdr, timeout=30).status_code == 401
    assert requests.get(mon["base"] + "/v1/monitoring", timeout=30).status_code == 401                      # reading needs the read key
    assert not mon["file"].exists()


def test_publish_rejects_bad_older_future_and_oversized_reports(mon):
    H = {"X-Monitoring-Publish-Key": mon["pub"], "Content-Type": "application/json"}
    put = lambda data: requests.put(mon["base"] + "/v1/monitoring/report", data=data, headers=H, timeout=30)
    assert put("not json").status_code == 422
    assert put(json.dumps({**report("ok"), "status": "GREEN"})).status_code == 422
    assert put(json.dumps(report("ok", age_hours=-1.0))).status_code == 422                                 # dated in the future
    assert put(b"x" * 1_100_000).status_code == 413
    publish(mon, "ok", age_hours=1.0)
    assert put(json.dumps(report("watch", age_hours=5.0))).status_code == 409                              # older than the stored report
    assert client(mon).monitoring()["status"] == "OK"                                                       # a rejected publish changes nothing


def test_unreadable_stored_report_gives_clear_503(mon, ui):
    mon["file"].write_text("{corrupt")
    e = err(client(mon).monitoring)
    assert e.kind == "unavailable" and "re-run the monitor" in e.user_message
    t = txt(ui("Monitoring")); assert "unreadable" in t and "Traceback" not in t


def test_wrong_read_key_is_an_auth_error(mon):
    publish(mon, "ok")
    assert err(client(mon, key="wrong").monitoring).kind == "auth"


# ---- Task 18's own verification script against the same live API --------------------------------------------------------------------------
def test_task18_verify_live_script_verifies_the_deployment(mon, tmp_path):
    script = Path(os.environ["CORTEXHR_API_REPO"]).parents[0] / "task18" / "verify_live.py"
    if not script.exists():
        pytest.skip("Task 18 verify_live.py not found next to the API repo")
    rep = tmp_path / "latest_report.json"; rep.write_text(json.dumps(report("watch")))
    env = {**os.environ, "CORTEXHR_API_URL": mon["base"], "CORTEXHR_API_KEY": mon["key"], "CORTEXHR_MONITORING_PUBLISH_KEY": mon["pub"]}
    p = subprocess.run([sys.executable, str(script), "--report", str(rep), "--publish"], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "VERIFIED" in p.stdout and "NOT VERIFIED" not in p.stdout, p.stdout + p.stderr
