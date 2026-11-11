import json

import httpx
import pytest

from cortexhr_task10.identity import ANONYMOUS, DemoUserDirectory, Identity, IdentityError, resolve_identity
from cortexhr_task10.policy import authorize, load_policies, placeholders
from cortexhr_task10.dialogue import REQUIRED_SLOTS
from cortexhr_task10.service_client import AiServiceClient, HRServiceClient, ServiceError
from cortexhr_task10.sessions import SessionIdError, SessionStore, validate_session_id
from support import RESPONSES, USERS

P = load_policies(RESPONSES)
SID = "A" * 24


# ---- sessions (#7, #23) ----
@pytest.mark.parametrize("bad", ["EMP-0188", "short", "a@b.example.com1234567", "x" * 65, "has space in it 123456", "semi;colon;123456789012", ""])
def test_session_id_format_is_validated(bad):
    with pytest.raises(SessionIdError):
        validate_session_id(bad)


def test_same_session_id_from_two_users_gives_two_isolated_sessions():
    s = SessionStore()
    a, ida, new_a = s.get_or_create("EMP-1001", SID)
    a.last_intent = "payslip_status"; a.slots["month"] = "2026-03"
    assert new_a and ida != SID                                      # ids are always server-issued
    a2, _, again = s.get_or_create("EMP-1001", ida)
    assert a2 is a and not again
    b, idb, new_b = s.get_or_create("EMP-1002", ida)                 # B replays A's real id
    assert new_b and idb != ida and b is not a and b.last_intent is None and b.slots == {}


def test_expired_and_unknown_sessions_are_replaced_safely():
    t = [0.0]
    s = SessionStore(ttl_seconds=10, clock=lambda: t[0])
    st, sid, _ = s.get_or_create("U", None); st.last_intent = "leave_balance"
    t[0] = 11
    fresh, sid2, new = s.get_or_create("U", sid)
    assert new and fresh.last_intent is None and sid2 != sid
    _, sid3, new3 = s.get_or_create("U", "Z" * 20)                   # well-formed but unknown
    assert new3


def test_session_store_is_bounded():
    s = SessionStore(max_sessions=3)
    for i in range(10):
        s.get_or_create(f"U{i}", None)
    assert len(s) <= 3


def test_session_state_holds_no_text_or_pay_fields():
    st, _, _ = SessionStore().get_or_create("U", None)
    assert set(vars(st)) == {"last_intent", "slots", "pending", "turns"}


# ---- identity ----
def test_identity_comes_from_the_server_side_table_only():
    users = DemoUserDirectory(USERS)
    assert resolve_identity(None, users) == ANONYMOUS and resolve_identity("  ", users) == ANONYMOUS
    assert resolve_identity("EMP-2001", users) == Identity("EMP-2001", "HR")
    assert resolve_identity("EMP-9001", users).role is None            # unrecognised role string -> unknown role
    for bad in ("EMP-0000", "EMP-1001\r\nX-Demo-User: EMP-2001", "../etc", "a" * 40):
        with pytest.raises(IdentityError):
            resolve_identity(bad, users)


# ---- authorization (#2, #3, #5, #8) ----
ROLES = [Identity("E", r) for r in ("EMPLOYEE", "HR", "ADMIN")]


@pytest.mark.parametrize("who", ROLES)
def test_salary_of_other_is_refused_for_every_role_and_target_flag(who):
    for target_is_self in (True, False):
        d = authorize(who, "salary_of_other", P, target_is_self=target_is_self)
        assert (d.allowed, d.code) == (False, "refuse_always")
    assert authorize(ANONYMOUS, "salary_of_other", P).code == "refuse_always"


def test_authorization_matrix_follows_chatbot_responses():
    emp, hr = Identity("E", "EMPLOYEE"), Identity("H", "HR")
    assert authorize(emp, "payslip_status", P).allowed and authorize(hr, "payslip_status", P).allowed
    for hr_only in ("salary_band_query", "attrition_risk_query", "headcount_forecast", "payroll_flags", "resume_score_request"):
        assert authorize(emp, hr_only, P).code == "role_not_allowed" and authorize(hr, hr_only, P).allowed
    assert authorize(ANONYMOUS, "greet", P).allowed and authorize(ANONYMOUS, "payslip_status", P).code == "login_required"
    assert authorize(Identity("X", None), "leave_balance", P).code == "unknown_role"
    assert authorize(emp, "no_such_intent", P).code == "unknown_intent" and authorize(emp, None, P).code == "unknown_intent"
    assert authorize(emp, "leave_balance", P, target_is_self=False).code == "other_person"


def test_a_modified_config_cannot_make_salary_of_other_answerable():
    from dataclasses import replace
    tampered = dict(P); tampered["salary_of_other"] = replace(P["salary_of_other"], call="GET /api/v1/payslips/me/{month}", allowed_roles=None)
    assert authorize(Identity("A", "ADMIN"), "salary_of_other", tampered).code == "refuse_always"     # enforced by NAME, not only by config


# ---- config consistency ----
def test_config_has_33_intents_and_required_slots_match_the_placeholders():
    assert len(P) == 33 and P["salary_of_other"].call == "REFUSE"
    names = {"month": "month", "department": "dept", "role": "role"}
    for intent, slots in REQUIRED_SLOTS.items():
        seen = placeholders(P[intent].call) | placeholders(P[intent].template)
        assert all(names[s] in seen for s in slots), intent


# ---- service client (#27 - #30) ----
def client(handler):
    return HRServiceClient(None, client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://hr.test"))


def test_header_is_the_authenticated_id_and_employee_param_is_always_me():
    seen = []
    c = client(lambda r: (seen.append((r.url.path, dict(r.url.params), r.headers.get("x-demo-user"))), httpx.Response(200, json={"cl": 1}))[1])
    c.leave_balance("EMP-1001")
    assert seen == [("/api/v1/leave/balance", {"employee": "me"}, "EMP-1001")]


@pytest.mark.parametrize("month", ["2026-3", "2026-13", "../x", "2026-03/../../employees", "", "March"])
def test_month_is_validated_before_it_goes_into_a_url(month):
    c = client(lambda r: pytest.fail("must not be called"))
    with pytest.raises(ServiceError) as e:
        c.payslip("EMP-1001", month)
    assert e.value.kind == "invalid"


def test_protected_endpoints_refuse_to_run_without_an_identity_and_reject_odd_ids():
    c = client(lambda r: pytest.fail("must not be called"))
    for call in (lambda: c.payslip(None, "2026-03"), lambda: c.employee_me(None), lambda: c.team("EMP 1;DROP")):
        with pytest.raises(ServiceError) as e:
            call()
        assert e.value.kind == "unauthorized"


@pytest.mark.parametrize("handler,kind", [
    (lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("t")), "timeout"),
    (lambda r: (_ for _ in ()).throw(httpx.ConnectError("c")), "unavailable"),
    (lambda r: httpx.Response(500, json={"detail": "SECRET-INTERNALS"}), "http"),
    (lambda r: httpx.Response(401), "unauthorized"), (lambda r: httpx.Response(404, json={"detail": "x"}), "not_found"),
    (lambda r: httpx.Response(200, content=b"<html>not json</html>"), "invalid"),
    (lambda r: httpx.Response(200, json=[1, 2]), "invalid")])
def test_failures_become_short_error_kinds_without_bodies(handler, kind):
    with pytest.raises(ServiceError) as e:
        client(handler).leave_balance("EMP-1001")
    assert e.value.kind == kind and "SECRET" not in str(e.value)


def test_no_base_url_means_unavailable_not_a_crash():
    with pytest.raises(ServiceError) as e:
        AiServiceClient(None).call("GET", "/ai/v1/forecast/headcount", "EMP-2001")
    assert e.value.kind == "unavailable"
