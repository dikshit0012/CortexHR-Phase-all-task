"""All 12 official dialogues, run through the full stack: REAL Task 9 classifier -> dialogue manager -> REAL mock_server.py code
(on synthetic mock_data) / recorded /ai/v1 stand-in. A dialogue passes only if EVERY turn meets EVERY check in dialogue_runner.py."""
import pytest

from dialogue_runner import load_dialogues, run_dialogue

DLGS = load_dialogues()


def test_all_twelve_official_dialogues_are_present():
    assert sorted(DLGS) == [f"DLG-{i:02d}" for i in range(1, 13)] and sum(len(v) for v in DLGS.values()) == 28


@pytest.mark.real_classifier
@pytest.mark.parametrize("dlg", sorted(DLGS))
def test_official_dialogue(env_real, dlg):
    recs = run_dialogue(env_real, dlg, DLGS[dlg])
    failures = [(r.turn, r.text, [k for k, v in r.checks.items() if not v], r.actual_intent, r.confidence) for r in recs if not r.passed]
    assert not failures, failures


@pytest.mark.real_classifier
def test_dlg05_refusal_then_own_payslip_never_touches_another_persons_data(env_real):
    recs = run_dialogue(env_real, "DLG-05", DLGS["DLG-05"])
    assert recs[0].observed_http == [] and recs[0].reply.startswith("Sorry, I cannot share another person's pay")
    assert recs[1].observed_http == ["GET /api/v1/payslips/me/2026-03"] and {h for _, _, h in env_real.hr.log} == {"EMP-1001"}
    assert "NETPAY-SENTINEL" not in recs[0].reply + recs[1].reply


def test_the_official_file_is_not_modified_by_the_tests():
    import hashlib
    from dialogue_runner import DIALOGUES
    official_sha256 = "4c35e12b9992520438e3d08c5ed536b1c9dd8f4c9f928ff24110e5fc7cccbefc"
    assert hashlib.sha256(DIALOGUES.read_bytes()).hexdigest() == official_sha256
