import json
import shutil

import pandas as pd
import pytest

from cortexhr_task9.faq import FaqError, load_faq, load_mapping, load_responder
from cortexhr_task9.paths import DEFAULT_FAQ, INTENT_FAQ_MAP, TRAIN_CSV
from cortexhr_task9.data import load_training, phrase_group


@pytest.fixture(scope="module")
def faq():
    return load_faq(DEFAULT_FAQ)


def test_faq_file_loads_with_real_content(faq):
    assert len(faq) == 30
    assert faq["FAQ-18"].answer == "30 days for L1-L2, 60 days for L3, 90 days for L4 and above."
    assert faq["FAQ-28"].answer == "payroll@znova-demo.example"
    assert all(e.answer and e.policy_source for e in faq.values())


def test_mapping_covers_every_model_intent_and_only_existing_faqs(model, responder, faq):
    assert set(responder.mapping) == set(model.classes)
    for entry in responder.mapping.values():
        for rule in entry.rules:
            assert all(i in faq for i in rule.faq_ids)
            assert rule.match_any                                       # no catch-all rules
    assert responder.reachable_faq_ids()                                # something is answerable


def test_answers_are_verbatim_faq_text(responder, faq):
    ans = responder.answer("wfh_policy", "what is the work from home policy")
    assert ans.reply == faq["FAQ-21"].answer and ans.faq_ids == ("FAQ-21",)
    ans = responder.answer("notice_period_policy", "how long is the notice period")
    assert ans.reply == faq["FAQ-18"].answer


def test_selection_is_deterministic_and_prefers_the_specific_rule(responder, faq):
    a = responder.answer("notice_period_policy", "can the notice period be bought out")
    assert a.faq_ids == ("FAQ-19",)                                      # buy-out beats the generic notice rule (veto)
    assert responder.answer("hr_policy_leave", "how many casual leaves are allowed").faq_ids == ("FAQ-01",)
    assert responder.answer("hr_policy_leave", "how many sick leaves are allowed").faq_ids == ("FAQ-02",)
    for _ in range(3):
        assert responder.answer("hr_policy_leave", "what is the leave policy").faq_ids == ("FAQ-01", "FAQ-02", "FAQ-03")


def test_generic_leave_question_combines_only_approved_text(responder, faq):
    reply = responder.answer("hr_policy_leave", "tell me about the leave policy").reply
    for fid in ("FAQ-01", "FAQ-02", "FAQ-03"):
        assert faq[fid].answer in reply and faq[fid].question in reply


@pytest.mark.parametrize("intent,message", [
    ("pf_info", "how do i activate uan"),                    # PF intent, but hr_faq has nothing about UAN
    ("pf_info", "how is lop calculated"),                    # 'calculated' alone must not trigger the PF answer
    ("payslip_date", "not my salary"),                       # a bare 'salary' is not a question about the pay date
    ("tax_declaration", "where to upload investment proofs"),
    ("notice_period_policy", "how do i resign"),
    ("working_hours", "are there flexible timings"),
    ("holiday_list", "list of company holidays"),
    ("reimbursement_policy", "reimbursement limits"),
    ("insurance_benefits", "insurance coverage amount"),
    ("probation_policy", "when will i be confirmed"),
    ("who_to_contact", "hr contact details"),                # the only approved contact is for PAYROLL
])
def test_questions_the_faq_does_not_cover_get_no_answer(responder, intent, message):
    assert responder.answer(intent, message) is None


@pytest.mark.parametrize("intent", ["greet", "thanks", "goodbye", "help", "chitchat", "out_of_scope", "leave_balance", "my_manager", "salary_of_other", "payslip_status"])
def test_intents_without_faq_rules_never_answer(responder, intent):
    assert not responder.has_rules(intent)
    assert responder.answer(intent, "leave salary payroll notice period") is None


def test_every_training_message_of_a_faq_intent_gets_either_its_own_topic_or_nothing(responder, faq):
    """Validation against REAL training text (true intents): no rule may answer a message with an FAQ from another topic."""
    topic_of = {"hr_policy_leave": "Leave", "working_hours": "Working time", "wfh_policy": "Policy", "payslip_date": "Pay",
                "pf_info": "Statutory", "tax_declaration": "Statutory", "notice_period_policy": "Exit", "probation_policy": "Policy",
                "reimbursement_policy": "Expenses", "who_to_contact": "Contacts", "insurance_benefits": "Benefits", "holiday_list": "Policy"}
    train = load_training(TRAIN_CSV)
    answered = 0
    for row in train[train["intent"].isin(topic_of)].itertuples():
        ans = responder.answer(row.intent, phrase_group(row.text))
        if ans:
            answered += 1
            assert all(faq[i].category == topic_of[row.intent] for i in ans.faq_ids), (row.text, ans)
    assert answered > 100


def test_payroll_contact_is_the_only_contact_ever_offered(responder, faq):
    for intent in ("payslip_status", "payroll_flags", "payslip_date"):
        assert responder.contact_note(intent) == f"{faq['FAQ-28'].question} {faq['FAQ-28'].answer}"
    for intent in ("leave_balance", "greet", "my_manager", "who_to_contact", "hr_policy_leave"):
        assert responder.contact_note(intent) is None


# ---- load-time validation ------------------------------------------------------------------------------------------
def _write_mapping(tmp_path, mutate):
    data = json.loads(INTENT_FAQ_MAP.read_text(encoding="utf-8"))
    mutate(data["intents"])
    p = tmp_path / "map.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_mapping_with_unknown_faq_id_is_rejected(tmp_path, faq, model):
    p = _write_mapping(tmp_path, lambda m: m["wfh_policy"]["rules"][0].__setitem__("faq_ids", ["FAQ-99"]))
    with pytest.raises(FaqError, match="unknown"):
        load_mapping(p, faq, model.classes)


def test_mapping_missing_an_intent_is_rejected(tmp_path, faq, model):
    p = _write_mapping(tmp_path, lambda m: m.pop("greet"))
    with pytest.raises(FaqError, match="disagree"):
        load_mapping(p, faq, model.classes)


def test_mapping_with_bad_regex_or_empty_rule_is_rejected(tmp_path, faq, model):
    p = _write_mapping(tmp_path, lambda m: m["wfh_policy"]["rules"][0].__setitem__("match_any", ["(unclosed"]))
    with pytest.raises(FaqError, match="regular expression"):
        load_mapping(p, faq, model.classes)
    p = _write_mapping(tmp_path, lambda m: m["wfh_policy"]["rules"][0].__setitem__("match_any", []))
    with pytest.raises(FaqError, match="catch-all"):
        load_mapping(p, faq, model.classes)


def test_faq_file_problems_are_rejected(tmp_path):
    with pytest.raises(FaqError, match="not found"):
        load_faq(tmp_path / "missing.csv")
    df = pd.read_csv(DEFAULT_FAQ, dtype=str, keep_default_na=False)
    dup = tmp_path / "dup.csv"
    pd.concat([df, df.iloc[[0]]]).to_csv(dup, index=False)
    with pytest.raises(FaqError, match="duplicate"):
        load_faq(dup)
    nocol = tmp_path / "nocol.csv"
    df.drop(columns=["answer"]).to_csv(nocol, index=False)
    with pytest.raises(FaqError, match="missing columns"):
        load_faq(nocol)
    empty = tmp_path / "empty_answer.csv"
    df2 = df.copy(); df2.loc[0, "answer"] = " "
    df2.to_csv(empty, index=False)
    with pytest.raises(FaqError, match="empty"):
        load_faq(empty)


# Which intent is supposed to serve each FAQ's OWN question (FAQ-26 is only reachable through the benefits overview; FAQ-08/10/11/12/
# 14/15/27/29 have no intent at all in the training data - see reports/faq_coverage.md).
OWN_QUESTION_INTENT = {"FAQ-01": "hr_policy_leave", "FAQ-02": "hr_policy_leave", "FAQ-03": "hr_policy_leave", "FAQ-04": "hr_policy_leave",
                       "FAQ-05": "hr_policy_leave", "FAQ-06": "working_hours", "FAQ-07": "working_hours", "FAQ-09": "payslip_date",
                       "FAQ-13": "pf_info", "FAQ-16": "tax_declaration", "FAQ-17": "tax_declaration", "FAQ-18": "notice_period_policy",
                       "FAQ-19": "notice_period_policy", "FAQ-20": "probation_policy", "FAQ-21": "wfh_policy", "FAQ-22": "reimbursement_policy",
                       "FAQ-23": "reimbursement_policy", "FAQ-24": "insurance_benefits", "FAQ-25": "insurance_benefits",
                       "FAQ-28": "who_to_contact", "FAQ-30": "holiday_list"}


@pytest.mark.parametrize("faq_id,intent", sorted(OWN_QUESTION_INTENT.items()))
def test_each_reachable_faq_is_returned_for_its_own_question(responder, faq, faq_id, intent):
    from cortexhr_task9.text_cleaning import clean_text
    ans = responder.answer(intent, clean_text(faq[faq_id].question))
    assert ans is not None and ans.faq_ids == (faq_id,), (faq[faq_id].question, ans)
    assert ans.reply == faq[faq_id].answer
