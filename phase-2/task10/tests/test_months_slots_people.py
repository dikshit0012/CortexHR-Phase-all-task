from datetime import date

import pytest

from cortexhr_task10.months import extract_month
from cortexhr_task10.people import EmployeeDirectory, EmployeeRecord, disambiguation_question
from cortexhr_task10.sensitive import detect_other_reference
from cortexhr_task10.slots import extract_department, extract_entities, extract_role, slot_only

REF = date(2026, 4, 15)


# ---- #15 / #16: month extraction and normalisation (deterministic: the reference date is a parameter) ----
@pytest.mark.parametrize("text,expected", [
    ("show my payslip for March", "2026-03"), ("January", "2026-01"), ("april", "2026-04"), ("Aug", "2025-08"),
    ("December", "2025-12"),                                   # bare month after the reference month -> previous year (rule documented)
    ("March 2026", "2026-03"), ("payslip for Feb 2026", "2026-02"), ("2026-02", "2026-02"), ("03/2026", "2026-03"),
    ("last month", "2026-03"), ("previous month", "2026-03"), ("prior month", "2026-03"), ("this month", "2026-04"), ("current month", "2026-04"),
    ("March 2025", "2025-03"),
])
def test_month_is_normalised_to_yyyy_mm(text, expected):
    r = extract_month(text, REF)
    assert (r.status, r.value) == ("ok", expected)


def test_relative_months_cross_the_year_boundary():
    assert extract_month("last month", date(2026, 1, 10)).value == "2025-12"
    assert extract_month("last month", date(2026, 3, 31)).value == "2026-02"


@pytest.mark.parametrize("text,reason", [("next month", "next_month"), ("March and April", "multiple"), ("payslip for 2027-01", "future"),
                                          ("May 2026", "future"), ("last month and this month", "multiple")])
def test_unusable_months_are_invalid_not_guessed(text, reason):
    r = extract_month(text, REF)
    assert (r.status, r.value, r.reason) == ("invalid", None, reason)


@pytest.mark.parametrize("text", ["may I see my payslip", "what is the leave policy", "show details", "my manager"])
def test_no_month_found_in_ordinary_text(text):
    assert extract_month(text, REF).status == "none"


def test_may_counts_only_in_a_month_context():
    assert extract_month("payslip for May", REF).value == "2025-05"
    assert extract_month("May", REF).value == "2025-05"


# ---- departments / roles / slot-only ----
def test_roles_win_over_departments_and_aliases_resolve():
    assert extract_role("what about Software Engineer") == "Software Engineer"
    assert extract_entities("Support Executive")["departments"] == [] and extract_role("Support Executive") == "Support Executive"
    assert extract_department("for Support only") == "Customer Support"
    assert extract_department("who is at risk in sales") == "Sales" and extract_department("in Narnia") is None


@pytest.mark.parametrize("text", ["and last month?", "what about April?", "and in Engineering?", "for Support only", "and sick leave?", "what about Software Engineer"])
def test_slot_only_messages(text):
    assert slot_only(text, REF)


def test_attendance_anomaly_followup_uses_conversation_context(mk):
    env = mk({"show my attendance for last month": ("attendance_summary", 0.99)})
    first = env.say("EMP-1001", "show my attendance for last month")

    result = env.say("EMP-1001", "any unusual days?", first.session_id)

    assert result.intent == "unusual_attendance"
    assert result.followup is True
    assert result.action == "answered"
    assert len(env.ai.log) == 1


@pytest.mark.parametrize("text", ["what about Rahul for March", "show my payslip for March", "and his?", "hello", "what about priya in March"])
def test_a_leftover_word_such_as_a_name_is_never_a_pure_slot_follow_up(text):
    assert not slot_only(text, REF)


# ---- employee-name handling ----
DIR = EmployeeDirectory([EmployeeRecord("E1", "Priya Nair", "Sales"), EmployeeRecord("E2", "Rahul Sharma", "Engineering"),
                         EmployeeRecord("E3", "Rahul Sharma", "Finance"), EmployeeRecord("E4", "Karthik Iyer", "Operations")])


def test_full_unique_name_is_found_case_insensitively():
    m = DIR.find("show attendance for priya nair")
    assert [(x.kind, x.status) for x in m] == [("full", "unique")]


def test_partial_name_is_flagged_as_partial():
    m = DIR.find("what about Karthik")
    assert m and m[0].kind == "partial" and m[0].status == "unique"


def test_duplicate_full_names_are_ambiguous_and_the_question_is_safe():
    m = DIR.find("show attendance for Rahul Sharma")
    assert m[0].status == "ambiguous" and len(m[0].matches) == 2
    q = disambiguation_question(m[0])
    assert "more than one employee" in q and "Engineering" in q and "Finance" in q
    for banned in ("E2", "E3", "salary", "payroll", "payslip", "@"):          # no ids, no pay, no contact details
        assert banned not in q


def test_unknown_names_are_not_in_the_directory_and_never_invented():
    assert DIR.find("show attendance for Zed Quux") == []


@pytest.mark.parametrize("text,strong", [
    ("what is Priya Nair salary", True), ("What about Rahul's salary?", True), ("salary of priya nair", True), ("how much does my colleague earn", True),
    ("how much does priya earn", True), ("show attendance for Rahul Sharma", True), ("and his?", True), ("can you share her payslip", True),
    ("show my payslip for last month", False), ("is my March payslip ready", False), ("suggest a salary band for Data Analyst", False),
    ("any unusual payroll this month", False), ("Where to get Form 16", False), ("who is at risk of leaving in Sales", False), ("hi", False),
    ("when is salary credited", False)])
def test_other_person_detection(text, strong):
    assert detect_other_reference(text).strong is strong


def test_weak_references_are_weak_and_pay_questions_about_groups_fail_closed():
    r = detect_other_reference("show their leave balance")
    assert r.weak and not r.strong                      # on its own a weak reference is not enough to refuse (manager tests cover pay + weak)
    assert detect_other_reference("average pay for employees").strong      # 'pay for <unknown word>' is treated as aimed at someone else
