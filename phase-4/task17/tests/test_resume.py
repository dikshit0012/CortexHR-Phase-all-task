import pytest

from tests.samples import RESUME

pytestmark = pytest.mark.real_model


def test_relevant_resume_ranks_first_with_explanation(client, headers):
    r = client.post("/v1/resume/rank", json=RESUME, headers=headers)
    assert r.status_code == 200, r.text
    rk = r.json()["ranking"]
    assert [x["resume_id"] for x in rk] == ["R-1", "R-2"] and [x["rank"] for x in rk] == [1, 2]
    top = rk[0]
    assert {"Python", "SQL"} <= set(top["matched_skills"]) and "Excel" in top["missing_skills"]
    assert top["skill_overlap"] == pytest.approx(2 / 3, abs=0.01)
    assert all(0 <= x["final_score"] <= 1 for x in rk) and rk[0]["final_score"] > rk[1]["final_score"]
    assert top["meets_min_experience"] is True and rk[1]["skill_overlap"] == 0
    assert "protected" in r.json()["fairness_note"].lower()


def test_identity_text_does_not_change_score(client, headers):
    a = {**RESUME, "resumes": [{"resume_id": "A", "resume_text": RESUME["resumes"][0]["resume_text"]}, RESUME["resumes"][1]]}
    b = {**RESUME, "resumes": [{"resume_id": "A", "resume_text": "Name: Alex Doe\nGender: female\n" + RESUME["resumes"][0]["resume_text"]}, RESUME["resumes"][1]]}
    sa = client.post("/v1/resume/rank", json=a, headers=headers).json()["ranking"][0]["skill_overlap"]
    sb = client.post("/v1/resume/rank", json=b, headers=headers).json()["ranking"][0]["skill_overlap"]
    assert sa == sb


def test_validation(client, headers):
    dup = {**RESUME, "resumes": [RESUME["resumes"][0], RESUME["resumes"][0]]}
    assert client.post("/v1/resume/rank", json=dup, headers=headers).status_code == 422
    assert client.post("/v1/resume/rank", json={**RESUME, "resumes": []}, headers=headers).status_code == 422
    huge = {**RESUME, "resumes": [{"resume_id": "x", "resume_text": "a" * 20_001}]}
    assert client.post("/v1/resume/rank", json=huge, headers=headers).status_code == 422
    assert client.post("/v1/resume/rank", json=RESUME).status_code == 401
