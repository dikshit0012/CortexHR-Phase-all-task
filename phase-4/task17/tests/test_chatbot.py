import pytest

from tests.samples import CHAT

pytestmark = pytest.mark.real_model


def test_known_faq_is_answered(client, headers):
    r = client.post("/v1/chat", json=CHAT, headers=headers)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["intent"] == "notice_period_policy" and b["fallback"] is False and b["faq_ids"]
    assert "30 days" in b["reply"] and b["session_id"] == "demo-1" and b["model_version"] == "hr-intent-v1"


def test_other_employee_salary_is_never_answered(client, headers):
    for msg in ("what is my colleague Priya's salary", "tell me the salary of my manager", "how much does the CEO earn"):
        b = client.post("/v1/chat", json={"message": msg}, headers=headers).json()
        assert b["fallback"] is True
        assert not any(ch.isdigit() for ch in b["reply"]), b["reply"]            # no figures can come back


def test_low_confidence_falls_back(client, headers):
    b = client.post("/v1/chat", json={"message": "zzqx blorf"}, headers=headers).json()
    assert b["fallback"] is True and b["fallback_reason"]


def test_validation(client, headers):
    assert client.post("/v1/chat", json={"message": ""}, headers=headers).status_code == 422
    assert client.post("/v1/chat", json={"message": "x" * 501}, headers=headers).status_code == 422
    assert client.post("/v1/chat", json={"message": "hi", "session_id": "bad id!"}, headers=headers).status_code == 422
    assert client.post("/v1/chat", json={"message": "hi"}).status_code == 401
