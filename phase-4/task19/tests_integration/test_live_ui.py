"""Every console page rendered by Streamlit against the REAL API (no mocked HTTP)."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from api_client import ApiClient
from components import uploads
from config import Config
from contracts import ATTRITION_FIELDS, ATTRITION_SAMPLE, PAYBAND_SAMPLE

pytestmark = pytest.mark.live
APP = str(Path(__file__).resolve().parents[1] / "app.py")
LABEL = "AI-generated, for human review"
PAGES = ["Overview", "Attrition Risks", "Anomalies", "HR Assistant", "Resume Screener", "Pay Bands", "Workforce Forecast", "Monitoring"]


@pytest.fixture
def ui(api, monkeypatch):
    monkeypatch.setenv("CORTEXHR_API_URL", api[0])
    monkeypatch.setenv("CORTEXHR_API_KEY", api[1])
    monkeypatch.setenv("CORTEXHR_TIMEOUT_SECONDS", "60")

    def run(page):
        at = AppTest.from_file(APP, default_timeout=90)
        at.session_state["page"] = page
        return at.run()
    return run


def txt(at):
    return " ".join(x.value for g in (at.markdown, at.error, at.info, at.warning, at.caption, at.success, at.title, at.header, at.subheader) for x in g)


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    return at.run()


def direct(api):
    return ApiClient(Config(api_url=api[0], api_key=api[1], timeout=60))


def test_all_eight_pages_render_without_error(ui):
    for p in PAGES:
        at = ui(p)
        assert not at.exception, p
        assert not at.error, (p, [e.value for e in at.error])


def test_api_key_never_appears_in_any_rendered_element(ui, api):
    for p in PAGES:
        at = ui(p)
        everything = repr(at)          # every element, widget value and label the page produced
        assert api[1] not in everything, p


def test_overview_shows_real_numbers(ui, api):
    at = ui("Overview"); t = txt(at)
    h = direct(api).health()
    assert f"{len(h['models_available'])} of {len(h['models'])}" in t and "Connected" in t and "Not available yet" in t
    assert "Data unavailable" in t and "not password-protected" in t


def test_attrition_single_matches_direct_api_call(ui, api):
    at = click(ui("Attrition Risks"), "Estimate attrition probability")
    p = direct(api).attrition_predict([{k: (float(v) if k in ATTRITION_FIELDS and ATTRITION_FIELDS[k]["type"] == "number" else v)
                                        for k, v in ATTRITION_SAMPLE.items()}])["results"][0]["attrition_probability_12m"]
    t = txt(at)
    assert f"{p:.1%}" in t and LABEL in t and "predicted" in t.lower() and not at.exception


def test_attrition_batch_csv_matches_direct_api_call(ui, api):
    rows = [{"record_ref": f"emp-{i}", **ATTRITION_SAMPLE, "tenure_months": 6 + 8 * i, "last_engagement_score": 1 + (i % 5)} for i in range(14)]
    csv = pd.DataFrame(rows).to_csv(index=False).encode()
    at = ui("Attrition Risks"); at.file_uploader[0].upload("batch.csv", csv, "text/csv"); at.run()
    at = click(at, "Score batch")
    assert not at.exception and not at.error
    shown = at.dataframe[0].value
    recs = uploads.table_to_records(uploads.read_csv_upload("batch.csv", csv), ATTRITION_FIELDS, extra_ok={"record_ref"})
    expect = direct(api).attrition_predict(recs)
    exp = {r["record_ref"]: (f"{r['attrition_probability_12m']:.1%}", r["risk_band"]) for r in expect["results"]}
    got = {row["Employee label"]: (row["Estimated probability"], row["Risk band"]) for _, row in shown.iterrows()}
    assert got == exp and {"High", "Medium", "Low"} >= {b for _, b in got.values()} and len(got) == 14
    assert LABEL in txt(at) and any(b.label == "Download results (CSV)" for b in at.get("download_button"))


def test_anomaly_pages_are_honest_when_detectors_unavailable(ui, api):
    models = direct(api).health()["models"]
    for kind, label in (("attendance", "Check attendance record"), ("payroll", "Check payroll record")):
        at = click(ui("Anomalies"), label)
        t = txt(at)
        if models[f"{kind}_anomaly"]["status"] == "loaded":
            assert LABEL in t
        else:
            assert "not available on the server right now" in t and LABEL not in t and "Traceback" not in t and not at.exception


def test_assistant_answers_and_refuses_salary_lookup(ui):
    at = ui("HR Assistant"); at.chat_input[0].set_value("What is the notice period?").run()
    t = txt(at)
    assert "60 days for L3" in t and LABEL in t and "notice_period_policy" in t
    at.chat_input[0].set_value("What is EMP-0188's salary?").run()
    t = txt(at)
    assert "could not answer this confidently" in t and "EMP-0188" in t       # echoed question only; no figure revealed
    assert not any(c.isdigit() for c in at.chat_message[-1].markdown[0].value)


def test_resume_screener_ranks_relevant_candidate_first(ui):
    at = ui("Resume Screener")
    at.text_input[0].set_value("Data Analyst"); at.text_input[1].set_value("Python, SQL, Excel")
    at.text_area[1].set_value("Retail supervisor with 6 years in customer service and team management\n---\n"
                              "Data analyst with 4 years of experience using Python, SQL and Tableau to build reports")
    at.file_uploader[0].upload("someone_private_name.txt", b"Python developer with SQL and Excel experience, 3 years", "text/plain")
    at = click(at, "Rank candidates")
    assert not at.exception and not at.error
    df = at.dataframe[0].value
    assert list(df["Rank"]) == [1, 2, 3] and set(df["Candidate"]) == {"Pasted resume 1", "Pasted resume 2", "someone_private_name.txt"}
    assert df.iloc[0]["Candidate"] != "Pasted resume 1" and df["Score"].is_monotonic_decreasing       # retail resume must not win
    t = txt(at)
    assert LABEL in t and "human review required" in t.lower()


def test_resume_screener_blocks_bad_input_before_api(ui):
    at = ui("Resume Screener"); at = click(at, "Rank candidates")
    assert any("job title" in w.value for w in at.warning)


def test_payband_matches_direct_api(ui, api):
    at = click(ui("Pay Bands"), "Get pay recommendation")
    expect = direct(api).payband(PAYBAND_SAMPLE)["predicted_salary_mid"]
    assert f"{expect:,}" in " ".join(m.value for m in at.metric) and LABEL in txt(at) and not at.exception


def test_forecast_matches_direct_api_for_two_departments(ui, api):
    for dept in ("ALL", "D03"):
        at = ui("Workforce Forecast"); at.selectbox[0].set_value(dept); at = click(at, "Load forecast")
        exp = direct(api).forecast(dept, 6)["forecast"]
        shown = at.dataframe[0].value
        assert len(shown) == 6 and list(shown["Expected headcount"]) == [p["expected_headcount"] for p in exp] and not at.exception
        assert LABEL in txt(at) and "not a guaranteed outcome" in txt(at)


def test_monitoring_page_reports_missing_route_and_real_health(ui, api):
    t = txt(ui("Monitoring")); h = direct(api).health()
    at = ui("Monitoring")
    assert f"{len(h['models_available'])} of {len(h['models'])}" in " ".join(m.value for m in at.metric)
    assert "does not expose the Task 18 monitoring endpoint yet" in t


def test_api_outage_is_friendly_on_every_page(monkeypatch):
    monkeypatch.setenv("CORTEXHR_API_URL", "http://127.0.0.1:9"); monkeypatch.setenv("CORTEXHR_API_KEY", "k")
    for p in ("Overview", "Monitoring"):
        at = AppTest.from_file(APP, default_timeout=60); at.session_state["page"] = p; at.run()
        assert not at.exception and "Unable to connect to CortexHR" in " ".join(e.value for e in at.error) and "Traceback" not in repr(at)
    at = AppTest.from_file(APP, default_timeout=60); at.session_state["page"] = "Pay Bands"; at.run()
    next(b for b in at.button if b.label == "Get pay recommendation").click(); at.run()
    assert "Unable to connect to CortexHR" in " ".join(e.value for e in at.error) and not at.exception
