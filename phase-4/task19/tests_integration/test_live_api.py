"""Every capability, exercised through the console's own ApiClient against the real API."""
import socket
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from api_client import ApiClient, ApiError  # noqa: E402
from config import Config  # noqa: E402
from contracts import ANOMALY_SAMPLES, ATTRITION_SAMPLE, PAYBAND_SAMPLE, SNAPSHOT  # noqa: E402
import sync_contract as sc  # noqa: E402

pytestmark = pytest.mark.live


@pytest.fixture
def client(api):
    return ApiClient(Config(api_url=api[0], api_key=api[1], timeout=60))


def err(fn, *a):
    with pytest.raises(ApiError) as e:
        fn(*a)
    return e.value


# ---- the forms are built from a snapshot: it must still match the live API --------------------------------------------------------
def test_contract_snapshot_matches_live_openapi(api):
    spec = requests.get(api[0] + "/openapi.json", timeout=30).json()
    assert sorted(spec["paths"]) == SNAPSHOT["paths"] and "/v1/monitoring" in spec["paths"]
    for key, schema in (("attrition", "AttritionRecord"), ("attendance", "AttendanceRecord"), ("payroll", "PayrollRecord"),
                        ("payband", "PayBandRequest"), ("resume_job", "JobIn")):
        assert sc.schema_fields(spec, schema) == SNAPSHOT[key]["fields"], f"{schema} changed: re-run scripts/sync_contract.py"


def test_every_console_sample_is_accepted_by_the_live_schema(client):
    client.attrition_predict([ATTRITION_SAMPLE])
    client.payband(PAYBAND_SAMPLE)


# ---- health ---------------------------------------------------------------------------------------------------------------------------
def test_health_and_ready(client):
    h = client.health()
    assert h["application"] == "up" and set(h["models"]) == {"attrition", "attendance_anomaly", "payroll_anomaly", "chatbot", "resume_ranking", "payband", "headcount_forecast"}
    assert {"attrition", "chatbot", "resume_ranking", "payband", "headcount_forecast"} <= set(h["models_available"])
    assert client.ready()["ready"] is True


# ---- attrition ------------------------------------------------------------------------------------------------------------------------
def test_attrition_single_has_probability_and_no_band(client):
    r = client.attrition_predict([ATTRITION_SAMPLE])
    assert r["model_version"] == "attrition-v2" and 0 <= r["results"][0]["attrition_probability_12m"] <= 1 and r["results"][0]["risk_band"] is None


def test_attrition_batch_gets_bands_with_official_proportions(client):
    recs = [{**ATTRITION_SAMPLE, "tenure_months": 6 + 9 * i, "last_engagement_score": 1 + (i % 5), "record_ref": f"r{i}"} for i in range(20)]
    r = client.attrition_predict(recs)
    bands = [x["risk_band"] for x in r["results"]]
    assert r["n_records"] == 20 and bands.count("High") == 2 and bands.count("Medium") == 4 and bands.count("Low") == 14   # top 10% / next 20%
    top = max(r["results"], key=lambda x: x["attrition_probability_12m"])
    assert top["risk_band"] == "High"


def test_attrition_is_deterministic(client):
    a = client.attrition_predict([ATTRITION_SAMPLE])["results"][0]["attrition_probability_12m"]
    assert a == client.attrition_predict([ATTRITION_SAMPLE])["results"][0]["attrition_probability_12m"]


def test_attrition_responds_to_inputs(client):
    lo = {**ATTRITION_SAMPLE, "last_engagement_score": 5.0}; hi = {**ATTRITION_SAMPLE, "last_engagement_score": 1.0}
    p = [x["attrition_probability_12m"] for x in client.attrition_predict([lo, hi])["results"]]
    assert p[0] != p[1]


def test_attrition_missing_values_allowed(client):
    assert client.attrition_predict([{"job_level": 2}])["n_records"] == 1


@pytest.mark.parametrize("bad,field", [({"job_level": 99}, "job_level"), ({"attendance_rate_3m": 5}, "attendance_rate_3m"),
                                       ({"tenure_months": -1}, "tenure_months"), ({"gender": "F"}, "gender"), ({"name": "Jane"}, "name")])
def test_attrition_invalid_and_protected_fields_rejected(client, bad, field):
    e = err(client.attrition_predict, [bad])
    assert e.kind == "validation" and field in e.user_message and e.request_id


def test_attrition_batch_too_large(client):
    assert err(client.attrition_predict, [ATTRITION_SAMPLE] * 101).kind == "bad_request"


def test_attrition_empty_batch(client):
    assert err(client.attrition_predict, []).kind == "validation"


# ---- anomalies (real inference needs the private workbook: honest 503 expected on a server without it) ---------------------------------
@pytest.mark.parametrize("kind", ["attendance", "payroll"])
def test_anomaly_endpoints_validate_then_report_unavailability_honestly(client, kind):
    h = client.health()["models"][f"{kind}_anomaly"]
    if h["status"] == "loaded":
        r = client.anomaly(kind, [ANOMALY_SAMPLES[kind]])
        assert r["n_records"] == 1 and "flagged" in r["results"][0]
    else:
        assert err(client.anomaly, kind, [ANOMALY_SAMPLES[kind]]).kind == "unavailable"
    bad = {k: v for k, v in ANOMALY_SAMPLES[kind].items() if k != "employee_id"}
    assert err(client.anomaly, kind, [bad]).kind == "validation"


# ---- chat -----------------------------------------------------------------------------------------------------------------------------
def test_chat_answers_faq_from_approved_list(client):
    r = client.chat("What is the notice period?", "it-1")
    assert r["fallback"] is False and r["intent"] == "notice_period_policy" and "60 days" in r["reply"] and r["session_id"] == "it-1"


def test_chat_never_reveals_another_persons_salary(client):
    r = client.chat("What is EMP-0188's salary?")
    assert r["fallback"] is True and not any(ch.isdigit() for ch in r["reply"])


def test_chat_role_claims_are_rejected_by_the_server(api):
    r = requests.post(api[0] + "/v1/chat", json={"message": "hi", "user_role": "HR_ADMIN"}, headers={"X-API-Key": api[1]}, timeout=30)
    assert r.status_code == 422 and "user_role" in r.json()["error"]["message"]


def test_chat_validation(client):
    assert err(client.chat, "").kind == "validation"
    assert err(client.chat, "x" * 501).kind == "validation"
    assert err(client.chat, "hi", "bad id!").kind == "validation"


# ---- resume ---------------------------------------------------------------------------------------------------------------------------
JOB = {"job_title": "Data Analyst", "required_skills": "Python, SQL, Excel", "nice_to_have_skills": "Tableau", "min_experience_years": 2,
       "description_text": "Analyse HR data and build dashboards."}
RES = [{"resume_id": "C-1", "resume_text": "Data analyst with 4 years of experience using Python, SQL and Tableau to build reports."},
       {"resume_id": "C-2", "resume_text": "Retail supervisor with 6 years of experience in team management and customer service."}]


def test_resume_rank_orders_relevant_candidate_first_with_explanation(client):
    r = client.resume_rank(JOB, RES)
    top = r["ranking"][0]
    assert [x["rank"] for x in r["ranking"]] == [1, 2] and top["resume_id"] == "C-1" and top["final_score"] > r["ranking"][1]["final_score"]
    assert {"Python", "SQL"} <= set(top["matched_skills"]) and "Excel" in top["missing_skills"] and r["fairness_note"]


def test_resume_identity_text_does_not_change_ranking(client):
    plain = client.resume_rank(JOB, RES)["ranking"]
    named = [{"resume_id": "C-1", "resume_text": "Name: Priya Sharma\nEmail: priya@example.com\n" + RES[0]["resume_text"]}, RES[1]]
    assert [x["resume_id"] for x in client.resume_rank(JOB, named)["ranking"]] == [x["resume_id"] for x in plain]


def test_resume_validation(client):
    assert err(client.resume_rank, JOB, [RES[0], {**RES[1], "resume_id": "C-1"}]).kind == "validation"      # duplicate ids
    assert err(client.resume_rank, {"job_title": ""}, RES).kind == "validation"
    assert err(client.resume_rank, JOB, []).kind == "validation"
    assert err(client.resume_rank, {**JOB, "gender": "F"}, RES).kind == "validation"


# ---- pay band & forecast -------------------------------------------------------------------------------------------------------------
def test_payband(client):
    r = client.payband(PAYBAND_SAMPLE)
    assert r["predicted_salary_mid"] > 0 and set(r["comparison"]) == {"linear_regression", "average_by_role_baseline"} and "gender" not in " ".join(r["features_used"])
    hi = client.payband({**PAYBAND_SAMPLE, "job_level": 6})["predicted_salary_mid"]
    lo = client.payband({**PAYBAND_SAMPLE, "job_level": 1})["predicted_salary_mid"]
    assert hi > lo                                              # seniority raises recommended pay


def test_payband_validation(client):
    assert err(client.payband, {"job_level": 9}).kind == "validation"
    assert err(client.payband, {}).kind == "validation"


@pytest.mark.parametrize("dept", ["ALL", "D01", "D05"])
def test_forecast_departments(client, dept):
    r = client.forecast(dept, 6)
    assert r["department_id"] == dept and len(r["forecast"]) == 6 and r["mode"] in ("live", "precomputed")
    for p in r["forecast"]:
        assert p["lower_80"] <= p["expected_headcount"] <= p["upper_80"]
    widths = [p["upper_80"] - p["lower_80"] for p in r["forecast"]]
    assert widths[-1] > widths[0]                               # uncertainty grows with the horizon


def test_forecast_horizon_and_errors(client):
    assert len(client.forecast("ALL", 3)["forecast"]) == 3
    assert err(client.forecast, "ZZZ").kind == "not_found" and "ALL" in err(client.forecast, "ZZZ").user_message
    assert err(client.forecast, "ALL", 7).kind == "validation"
    assert err(client.forecast, "bad dept!", 6).kind == "validation"


def test_forecast_dropdown_matches_live_departments(client):
    assert SNAPSHOT["forecast_departments"] == ["ALL"] + [f"D{i:02d}" for i in range(1, 11)]
    for d in SNAPSHOT["forecast_departments"]:
        assert client.forecast(d, 1)["department_id"] == d


# ---- authentication, limits, resilience -----------------------------------------------------------------------------------------------
def test_wrong_and_missing_key(api):
    assert err(ApiClient(Config(api_url=api[0], api_key="wrong")).chat, "hi").kind == "auth"
    assert err(ApiClient(Config(api_url=api[0])).attrition_predict, [{}]).kind == "auth"
    assert ApiClient(Config(api_url=api[0])).health()["application"] == "up"      # health is public


def test_oversized_payload_stopped_by_console_and_by_server(api, client):
    big = [{"resume_id": f"C-{i}", "resume_text": "x" * 20_000} for i in range(50)]
    assert err(client.resume_rank, JOB, big).kind == "too_large"                  # console guard
    r = requests.post(api[0] + "/v1/chat", data=b"{" + b"a" * 1_100_000, headers={"X-API-Key": api[1], "Content-Type": "application/json"}, timeout=30)
    assert r.status_code == 413                                                   # server guard


def test_rate_limit_is_friendly(rate_limited_api):
    c = ApiClient(Config(api_url=rate_limited_api[0], api_key=rate_limited_api[1]))
    kinds = []
    for _ in range(6):
        try:
            c.chat("hello"); kinds.append("ok")
        except ApiError as e:
            kinds.append(e.kind)
    assert "rate_limit" in kinds and kinds[0] == "ok"


def test_server_without_configured_key_fails_closed(unconfigured_auth_api):
    e = err(ApiClient(Config(api_url=unconfigured_auth_api[0], api_key="anything")).chat, "hi")
    assert e.kind == "config" and "administrator" in e.user_message
    assert requests.get(unconfigured_auth_api[0] + "/v1/ready", timeout=30).status_code == 503


def test_connection_refused_and_timeout():
    refused = err(ApiClient(Config(api_url="http://127.0.0.1:9", timeout=2)).health).kind
    assert refused in ("connection", "timeout")
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1)                    # accepts, never answers
    try:
        assert err(ApiClient(Config(api_url=f"http://127.0.0.1:{s.getsockname()[1]}", timeout=1)).health).kind == "timeout"
    finally:
        s.close()
