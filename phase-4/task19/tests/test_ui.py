"""Drive the real Streamlit pages. HTTP is mocked ONLY here, with payloads captured from the real API; the real-API runs are in
tests_integration/."""
from pathlib import Path

import pandas as pd
import pytest
import responses
from streamlit.testing.v1 import AppTest

from contracts import ATTENDANCE_SAMPLE, ATTRITION_SAMPLE, PAYROLL_SAMPLE
from tests.conftest import FIX, real

APP = str(Path(__file__).resolve().parents[1] / "app.py")
BASE = "https://api.test"
LABEL = "AI-generated, for human review"
HEALTH = {"application": "up", "api_version": "1.0.0", "uptime_seconds": 5, "environment": "test",
          "models_available": ["attrition", "chatbot", "headcount_forecast"], "models_unavailable": ["attendance_anomaly"],
          "models": {"attrition": {"status": "loaded", "version": "v2", "mode": None}, "chatbot": {"status": "loaded", "version": "v1", "mode": None},
                     "headcount_forecast": {"status": "loaded", "version": "v1", "mode": "precomputed"},
                     "attendance_anomaly": {"status": "unavailable", "version": "v1", "mode": None}}}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("CORTEXHR_API_URL", BASE)
    monkeypatch.setenv("CORTEXHR_API_KEY", "k")


def run(page="Overview", **_):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["page"] = page
    return at.run()


def txt(at):
    return " ".join(x.value for g in (at.markdown, at.error, at.info, at.warning, at.caption, at.success) for x in g)


def api(routes):
    for (m, p), (status, body) in routes.items():
        getattr(responses, m)(f"{BASE}{p}", json=body, status=status)


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    return at.run()


# ---- configuration and gate ---------------------------------------------------------------------------------------------------
def test_missing_config(monkeypatch):
    monkeypatch.delenv("CORTEXHR_API_URL")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("not configured" in e.value for e in at.error) and not at.exception


def test_invalid_url_config(monkeypatch):
    monkeypatch.setenv("CORTEXHR_API_URL", "ftp://x")
    assert any("not configured" in e.value for e in AppTest.from_file(APP, default_timeout=30).run().error)


@responses.activate
def test_gate_blocks_until_correct_password(monkeypatch):
    monkeypatch.setenv("CORTEXHR_CONSOLE_PASSWORD", "open-sesame-123")
    api({("get", "/v1/health"): (200, HEALTH), ("get", "/v1/ready"): (200, {"ready": True}), ("get", "/v1/monitoring"): (404, {})})
    at = AppTest.from_file(APP, default_timeout=30).run()
    from streamlit.proto.TextInput_pb2 import TextInput
    assert at.text_input[0].proto.type == TextInput.PASSWORD and "Workforce Intelligence Console" in at.title[0].value
    assert not responses.calls                      # nothing reaches the API before sign-in
    at.text_input[0].set_value("wrong").run(); at.button[0].click().run()
    assert any("Incorrect" in e.value for e in at.error) and not responses.calls
    at.text_input[0].set_value("open-sesame-123"); at.button[0].click().run()
    assert "Start here" in txt(at) and not at.exception


@responses.activate
def test_gate_lockout_after_five_failures(monkeypatch):
    monkeypatch.setenv("CORTEXHR_CONSOLE_PASSWORD", "pw")
    at = AppTest.from_file(APP, default_timeout=30).run()
    for _ in range(5):
        at.text_input[0].set_value("bad"); at.button[0].click().run()
    assert any("Too many incorrect attempts" in e.value for e in at.error)


# ---- overview ----------------------------------------------------------------------------------------------------------------------
@responses.activate
def test_overview_real_data_only_and_honest_gaps():
    api({("get", "/v1/health"): (200, HEALTH), ("get", "/v1/ready"): (200, {"ready": True, "blocking": [], "degraded": []}),
           ("get", "/v1/monitoring"): (404, {"error": {"code": "not_found", "message": "x"}}),
           ("post", "/v1/forecast/headcount"): (200, real("forecast"))})
    at = run("Overview"); t = txt(at)
    assert "Connected" in t and "3 of 4" in t and t.count("Data unavailable") >= 3 and "Not available yet" in t
    assert "444" in t and "not password-protected" in t and not at.exception


@responses.activate
def test_overview_api_down():
    t = txt(run("Overview"))     # no mocks registered -> connection error
    assert "Unable to connect to CortexHR" in t and "Unavailable" in t


@responses.activate
def test_overview_start_here_button_navigates():
    api({("get", "/v1/health"): (200, HEALTH), ("get", "/v1/ready"): (200, {"ready": True}), ("get", "/v1/monitoring"): (404, {}),
           ("post", "/v1/forecast/headcount"): (200, real("forecast"))})
    at = run("Overview"); at = click(at, "Open Attrition Risks →")
    assert at.session_state["page"] == "Attrition Risks" and at.header[0].value == "Attrition Risks"


# ---- attrition ---------------------------------------------------------------------------------------------------------------------
@responses.activate
def test_risk_single_success_wording_and_label():
    api({("post", "/v1/attrition/predict"): (200, real("attrition1"))})
    at = click(run("Attrition Risks"), "Estimate attrition probability"); t = txt(at)
    assert LABEL in t and "19.9%" in t and "predicted" in t.lower() and "does not establish that an employee will leave" in t
    assert "not available for a single employee" in t and "Human review required" in t and not at.exception
    sent = responses.calls[-1].request.body
    assert b'"job_level": 3.0' in sent and b"employee" not in sent.lower().replace(b"employment_type", b"")


@responses.activate
def test_risk_errors_are_friendly_and_show_no_result():
    api({("post", "/v1/attrition/predict"): (503, {"error": {"code": "model_unavailable", "message": "Traceback secret", "request_id": "abc123"}})})
    at = click(run("Attrition Risks"), "Estimate attrition probability"); t = txt(at)
    assert "not available on the server right now" in t and "abc123" in t and "Traceback" not in t and LABEL not in t


@responses.activate
def test_risk_batch_csv_upload_shows_bands():
    res = {"model_version": "attrition-v2", "n_records": 12, "risk_band_note": "n", "results": [
        {"record_ref": f"b{i}", "attrition_probability_12m": 0.05 * (i + 1), "risk_band": "High" if i > 9 else "Medium" if i > 7 else "Low"} for i in range(12)]}
    api({("post", "/v1/attrition/predict"): (200, res)})
    csv = pd.DataFrame([{"record_ref": f"b{i}", **ATTRITION_SAMPLE} for i in range(12)]).to_csv(index=False).encode()
    at = run("Attrition Risks"); at.file_uploader[0].upload("emps.csv", csv, "text/csv"); at.run()
    at = click(at, "Score batch"); t = txt(at)
    assert LABEL in t and "Employees scored:** 12" in t and not at.exception and len(at.dataframe) == 1
    assert len(responses.calls[-1].request.body) > 1000


@responses.activate
def test_risk_batch_rejects_identifiers_and_never_calls_api():
    csv = pd.DataFrame([{"name": "Jane Roe", **ATTRITION_SAMPLE}]).to_csv(index=False).encode()
    at = run("Attrition Risks"); at.file_uploader[0].upload("emps.csv", csv, "text/csv"); at.run()
    at = click(at, "Score batch")
    assert any("personal identifiers" in e.value for e in at.error) and not responses.calls


@responses.activate
def test_risk_batch_rejects_wrong_file_type():
    at = run("Attrition Risks"); at.file_uploader[0].upload("emps.xlsx", b"PK\x03\x04", "application/zip"); at.run()
    assert not [b for b in at.button if b.label == "Score batch"] and not responses.calls   # the uploader itself refuses non-CSV files


# ---- anomalies ---------------------------------------------------------------------------------------------------------------------
ANOM = {"model_version": "attendance-anomaly-v1", "n_records": 1, "n_flagged": 1, "note": "n", "results": [
    {"result_id": "AN-0001", "source_dataset": "attendance", "record_id": "AF-EX-0001", "employee_id": "EMP-0002", "period": "2026-10-01",
     "anomaly_score": 0.93, "flagged": "Y", "reason_code": "LONG_UNPAID_STREAK", "reason_text": "21 unpaid days in a row", "money_at_risk": 1200.5,
     "model_version": "attendance-anomaly-v1", "scored_at": "2026-11-19 10:00:00"}]}


@responses.activate
def test_anomaly_attendance_success_fields_and_label():
    api({("post", "/v1/anomaly/attendance"): (200, ANOM)})
    at = click(run("Anomalies"), "Check attendance record"); t = txt(at)
    assert LABEL in t and "21 unpaid days in a row" in t and "not proof" in t and not at.exception
    assert any(m.label.startswith("Money at risk") and m.value == "1,200.50" for m in at.metric)
    assert b"EMP-0002" in responses.calls[-1].request.body


@responses.activate
def test_anomaly_503_is_honest_not_a_result():
    api({("post", "/v1/anomaly/attendance"): (503, real("anomaly_503"))})
    at = click(run("Anomalies"), "Check attendance record"); t = txt(at)
    assert "not available on the server right now" in t and "CORTEXHR_WORKBOOK" not in t and LABEL not in t


@responses.activate
def test_anomaly_payroll_posts_to_payroll_route():
    api({("post", "/v1/anomaly/payroll"): (200, {**ANOM, "model_version": "payroll-anomaly-v1"})})
    at = run("Anomalies")
    at = click(at, "Check payroll record")
    assert responses.calls[-1].request.url.endswith("/v1/anomaly/payroll") and b"PI-EX-0001" in responses.calls[-1].request.body


# ---- assistant ----------------------------------------------------------------------------------------------------------------------
@responses.activate
def test_chat_answer_label_and_no_role_sent():
    api({("post", "/v1/chat"): (200, real("chat_faq"))})
    at = run("HR Assistant"); assert "Verify important HR decisions with an authorized human" in txt(at)
    at.chat_input[0].set_value("What is the notice period?").run(); t = txt(at)
    assert "30 days for L1-L2" in t and LABEL in t and "notice_period_policy" in t and not at.exception
    assert b"role" not in responses.calls[-1].request.body


@responses.activate
def test_chat_fallback_shown_plainly():
    api({("post", "/v1/chat"): (200, real("chat_salary_other"))})
    at = run("HR Assistant"); at.chat_input[0].set_value("salary of EMP-0188?").run()
    assert "could not answer this confidently" in txt(at) and "not sure I understood" in txt(at)


@responses.activate
def test_chat_api_down_is_friendly():
    at = run("HR Assistant"); at.chat_input[0].set_value("hi").run()
    assert "Unable to connect to CortexHR" in txt(at) and not at.exception


# ---- resume --------------------------------------------------------------------------------------------------------------------------
@responses.activate
def test_resume_requires_title_and_resume():
    at = click(run("Resume Screener"), "Rank candidates")
    assert any("job title" in w.value for w in at.warning) and not responses.calls


@responses.activate
def test_resume_rank_success_with_paste_and_txt_and_privacy():
    api({("post", "/v1/resume/rank"): (200, real("resume"))})
    at = run("Resume Screener")
    at.text_input[0].set_value("Data Analyst")
    at.text_area[1].set_value("Python SQL analyst, 4 years\n---\nRetail supervisor")
    at.file_uploader[0].upload("jane_roe_cv.txt", b"Data analyst with Python", "text/plain")
    at = click(at, "Rank candidates"); t = txt(at)
    sent = responses.calls[-1].request.body
    assert LABEL in t and "human review required" in t.lower() and "Do not reject candidates automatically" in t and not at.exception
    assert b"jane_roe" not in sent and b'"resume_id": "C-1"' in sent and b'"resume_id": "C-3"' in sent   # filenames never leave the browser


@responses.activate
def test_resume_rejects_pdf_without_calling_api():
    at = run("Resume Screener")
    at.text_input[0].set_value("Analyst"); at.file_uploader[0].upload("cv.pdf", b"%PDF", "application/pdf")
    at = click(at, "Rank candidates")
    assert not responses.calls


# ---- pay bands & forecast ------------------------------------------------------------------------------------------------------------
@responses.activate
def test_payband_success():
    api({("post", "/v1/payband/analyze"): (200, real("payband"))})
    at = click(run("Pay Bands"), "Get pay recommendation"); t = txt(at)
    assert LABEL in t and "calibration file not configured" in t and "1,469,931" in " ".join(m.value for m in at.metric) and not at.exception


@responses.activate
def test_forecast_success_chart_table_events_label():
    api({("post", "/v1/forecast/headcount"): (200, real("forecast"))})
    at = click(run("Workforce Forecast"), "Load forecast"); t = txt(at)
    assert LABEL in t and "PRECOMPUTED" in t and "not a guaranteed outcome" in t and "no historical line is drawn" in t and not at.exception
    assert len(at.dataframe) >= 1


@responses.activate
def test_forecast_unknown_department_message_comes_from_api():
    api({("post", "/v1/forecast/headcount"): (404, real("forecast_bad_dept"))})
    at = click(run("Workforce Forecast"), "Load forecast")
    assert any("No forecast exists" in i.value for i in at.info) and LABEL not in txt(at)


# ---- monitoring (bodies captured from the REAL patched API after publishing real monitor.py reports) ----------------------------------
MON = {k: v["body"] for k, v in __import__("json").loads((Path(__file__).parent / "fixtures" / "monitoring_samples.json").read_text()).items()}
HEALTH_RDY = {("get", "/v1/health"): (200, HEALTH), ("get", "/v1/ready"): (200, {"ready": True, "blocking": [], "degraded": []})}


def mon_page(name):
    api({**HEALTH_RDY, ("get", "/v1/monitoring"): (200, MON[name])})
    return run("Monitoring")


@responses.activate
def test_monitoring_ok_report():
    at = mon_page("ok"); t = txt(at)
    assert "No feature shows drift" in t and LABEL in t and "not proof of discrimination" in t and not at.exception and not at.error
    assert ">OK<" in t and "outdated" not in t and "Twin-resume fairness (Task 12)" in t and "Pay-gap fairness (Task 15)" in t
    assert "Why the status is not OK" not in t


@responses.activate
def test_monitoring_watch_names_the_drifting_feature():
    at = mon_page("watch"); t = txt(at)
    assert "WATCH" in t and "no action is required yet" in t
    assert any("work_mode" in " ".join(map(str, df.value.values.flatten())) for df in at.dataframe)


@responses.activate
def test_monitoring_drift_alert_lists_all_drifting_features():
    at = mon_page("alert_drift"); t = txt(at)
    flat = " ".join(" ".join(map(str, df.value.values.flatten())) for df in at.dataframe)
    assert "ALERT" in t and "Escalate" in t and sum(1 for r in MON["alert_drift"]["reasons"] if r["item"] in flat) == 5


@responses.activate
def test_monitoring_fairness_alert_is_visible_even_if_drift_is_clean():
    at = mon_page("alert_fairness"); t = txt(at)
    assert "ALERT" in t and "Twin-resume fairness" in t and ">ALERT<" in t.replace(" ", "") or "ALERT" in t
    assert "Fairness" in " ".join(" ".join(map(str, df.value.values.flatten())) for df in at.dataframe)


@responses.activate
def test_monitoring_missing_evidence_is_not_presented_as_healthy():
    at = mon_page("alert_missing"); t = txt(at)
    assert "could not be evaluated" in t and "not a pass" in t and "ALERT" in t and "NOT EVALUATED" in t
    assert "No feature shows drift" not in t and "Missing evidence" in " ".join(" ".join(map(str, df.value.values.flatten())) for df in at.dataframe)


@responses.activate
def test_monitoring_stale_report_is_flagged_not_trusted():
    at = mon_page("stale_ok"); t = txt(at)
    assert "outdated" in t and "100 hours old" in t and "older than the allowed age" in t


@responses.activate
def test_monitoring_failed_run_warns_status_unreliable():
    bad = {**MON["ok"], "execution_ok": False, "execution_status": "FAILED"}
    api({**HEALTH_RDY, ("get", "/v1/monitoring"): (200, bad)})
    assert "monitoring run itself failed" in txt(run("Monitoring"))


@responses.activate
def test_monitoring_no_report_yet_and_unreadable_report():
    api({**HEALTH_RDY, ("get", "/v1/monitoring"): (404, {"error": {"code": "monitoring_report_not_found", "message": "No monitoring report has been published to this server yet.", "request_id": "r"}})})
    t = txt(run("Monitoring")); assert "No monitoring report has been published" in t and "no status is shown" in t
    responses.replace(responses.GET, f"{BASE}/v1/monitoring", status=503, json={"error": {"code": "monitoring_report_invalid", "message": "internal path /srv/x", "request_id": "r2"}})
    t = txt(run("Monitoring")); assert "unreadable" in t and "/srv/x" not in t and "r2" in t


@responses.activate
def test_monitoring_missing_route_is_not_treated_as_no_report():
    api({**HEALTH_RDY, ("get", "/v1/monitoring"): (404, {"error": {"code": "not_found", "message": "Not Found", "request_id": "r"}})})
    t = txt(run("Monitoring"))
    assert "does not expose the Task 18 monitoring endpoint yet" in t
    assert "No monitoring report has been published" not in t


@responses.activate
def test_monitoring_malformed_response_rejected():
    api({**HEALTH_RDY, ("get", "/v1/monitoring"): (200, {**MON["ok"], "status": "GREEN"})})
    t = txt(run("Monitoring")); assert "unexpected monitoring response" in t and "GREEN" not in t


@responses.activate
def test_overview_monitoring_card_shows_real_status_and_staleness():
    for name, expect in (("watch", "WATCH"), ("alert_drift", "ALERT"), ("stale_ok", "OK · outdated")):
        api({**HEALTH_RDY, ("get", "/v1/monitoring"): (200, MON[name]), ("post", "/v1/forecast/headcount"): (200, real("forecast"))})
        t = txt(run("Overview")); assert expect in t and "Not available yet" not in t, name
        responses.reset()


def test_console_never_publishes_or_triggers_monitoring():
    root = Path(__file__).parents[1]
    app_files = [p for p in root.rglob("*.py") if not any(part.startswith("tests") or part == "scripts" for part in p.relative_to(root).parts)]
    assert {f.name for f in app_files} >= {"app.py", "api_client.py", "monitoring.py", "overview.py"}   # the scan really covers the app
    src = " ".join(p.read_text(encoding="utf-8") for p in app_files)
    assert "Publish-Key" not in src and "PUBLISH" not in src and '"PUT"' not in src and "monitor.py" not in src
