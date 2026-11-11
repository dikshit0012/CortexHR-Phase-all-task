import logging

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

import cortexhr_task9.intent_model as intent_model
from cortexhr_task9.chat_service import (ANSWERED, EMPTY, ERROR, GENERIC_FALLBACK, LOW_CONFIDENCE, NO_ANSWER, NO_ANSWER_FALLBACK,
                                         OUT_OF_SCOPE, ChatService, load_chat_service)
from cortexhr_task9.data import load_training
from cortexhr_task9.faq import load_faq
from cortexhr_task9.paths import DEFAULT_FAQ, ENV_FAQ, ENV_MODEL, ENV_THRESHOLD, TRAIN_CSV
from conftest import FakeModel

FAQ = load_faq(DEFAULT_FAQ)
# Hand-written probes (NOT from the training or test files): unsupported, off-topic or HR-sounding questions the FAQ cannot answer.
UNSUPPORTED = ["what is the capital of France", "can I bring my dog to the office", "how do I reset my laptop password",
               "asdfgh qwerty", "123456", "what is the dress code", "how is overtime paid", "when is the performance bonus paid",
               "is smoking allowed in the office", "who is covered by ESI", "how do I update my bank details", "how is LOP calculated",
               "what is professional tax", "can I get a refund for my flight", "tell me a joke"]


def test_known_question_gets_the_approved_faq_answer(service):
    r = service.reply("what is the work from home policy")
    assert (r.intent, r.fallback, r.reason) == ("wfh_policy", False, ANSWERED)
    assert r.reply == FAQ["FAQ-21"].answer and r.faq_ids == ("FAQ-21",)
    assert r.confidence >= service.threshold


def test_messy_informal_question_is_still_answered(service):
    r = service.reply("plesae  whats the NOTICE period??")
    assert r.fallback is False and r.intent == "notice_period_policy" and r.reply == FAQ["FAQ-18"].answer


def test_low_confidence_triggers_fallback_without_an_intent(fake_service):
    r = fake_service(confidence=0.30).reply("what is the work from home policy")
    assert (r.intent, r.confidence, r.fallback, r.reason) == (None, 0.3, True, LOW_CONFIDENCE)
    assert r.reply == GENERIC_FALLBACK and "@" not in r.reply                      # no contact detail in the generic fallback


def test_threshold_boundary_is_inclusive(fake_service):
    assert fake_service(confidence=0.64, threshold=0.64).reply("work from home").fallback is False
    assert fake_service(confidence=0.6399, threshold=0.64).reply("work from home").reason == LOW_CONFIDENCE


@pytest.mark.parametrize("message", ["", " ", "   \t\n  ", None])
def test_empty_or_whitespace_message_is_handled_safely(service, message):
    r = service.reply(message)
    assert (r.intent, r.confidence, r.fallback, r.reason, r.reply) == (None, None, True, EMPTY, GENERIC_FALLBACK)


def test_out_of_scope_prediction_falls_back(fake_service):
    r = fake_service(intent="out_of_scope", confidence=0.95).reply("tell me a joke")
    assert (r.intent, r.fallback, r.reason, r.reply) == ("out_of_scope", True, OUT_OF_SCOPE, GENERIC_FALLBACK)


@pytest.mark.parametrize("intent", ["leave_balance", "my_manager", "greet", "salary_of_other", "apply_leave_howto"])
def test_intent_without_faq_mapping_triggers_fallback(fake_service, intent):
    r = fake_service(intent=intent, confidence=0.95).reply("anything")
    assert (r.intent, r.fallback, r.reason) == (intent, True, NO_ANSWER)
    assert r.reply == NO_ANSWER_FALLBACK and r.faq_ids == ()


def test_recognised_intent_with_uncovered_subquestion_falls_back(fake_service):
    r = fake_service(intent="pf_info", confidence=0.95).reply("how do i activate uan")
    assert r.fallback is True and r.reason == NO_ANSWER and r.intent == "pf_info"
    assert FAQ["FAQ-13"].answer not in r.reply                                       # the unrelated PF answer must not leak in


def test_only_the_approved_payroll_contact_is_ever_shown(fake_service):
    r = fake_service(intent="payslip_status", confidence=0.95).reply("show my payslip for june")
    assert r.fallback and r.reply.endswith(FAQ["FAQ-28"].answer) and r.reply.startswith(NO_ANSWER_FALLBACK)
    other = fake_service(intent="leave_balance", confidence=0.95).reply("leave balance")
    assert "@" not in other.reply


@pytest.mark.parametrize("message", UNSUPPORTED)
def test_unsupported_questions_never_receive_an_invented_hr_answer(service, message):
    r = service.reply(message)
    assert r.fallback is True, (message, r)
    assert r.reply in (GENERIC_FALLBACK, NO_ANSWER_FALLBACK)
    assert not any(a.answer in r.reply for a in FAQ.values() if a.faq_id != "FAQ-28")


def test_confidence_is_always_a_valid_probability(service):
    texts = list(load_training(TRAIN_CSV)["text"].sample(120, random_state=3)) + UNSUPPORTED + ["", "   "]
    for t in texts:
        r = service.reply(t)
        assert r.confidence is None or 0.0 <= r.confidence <= 1.0
        assert (r.intent is None) or r.confidence is not None
        assert r.fallback == (r.reason != ANSWERED)
        assert r.reply.strip()


def test_inference_errors_do_not_expose_internals(fake_service, caplog):
    secret = RuntimeError("boom /home/secret/model.pkl password=hunter2")
    with caplog.at_level(logging.ERROR, logger="cortexhr.chat"):
        r = fake_service(error=secret).reply("what is the notice period")
    assert (r.intent, r.confidence, r.fallback, r.reason, r.reply) == (None, None, True, ERROR, GENERIC_FALLBACK)
    assert "hunter2" not in repr(r) and "/home/secret" not in repr(r)
    assert "hunter2" not in caplog.text and "RuntimeError" in caplog.text                  # the log has the exception TYPE only


def test_invalid_model_output_becomes_a_fallback(service, monkeypatch):
    monkeypatch.setattr(type(service.model.pipeline), "predict_proba", lambda self, X: np.full((1, 33), np.nan))
    r = service.reply("what is the notice period")
    assert r.reason == ERROR and r.fallback and r.intent is None


def test_replying_never_trains_or_reloads(service, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("fit / load must not be called while answering")
    monkeypatch.setattr(Pipeline, "fit", boom)
    monkeypatch.setattr(intent_model.joblib, "load", boom)
    pipeline_before = service.model.pipeline
    for _ in range(5):
        assert service.reply("what is the work from home policy").fallback is False
    assert service.model.pipeline is pipeline_before


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_threshold_must_be_strictly_between_0_and_1(model, responder, bad):
    with pytest.raises(ValueError):
        ChatService(model, responder, threshold=bad)


def test_threshold_is_configurable_through_the_environment(artifact_path):
    env = {ENV_MODEL: str(artifact_path), ENV_FAQ: str(DEFAULT_FAQ)}
    assert load_chat_service(env).threshold == load_chat_service(env).model.threshold
    assert load_chat_service({**env, ENV_THRESHOLD: "0.9"}).threshold == 0.9
    with pytest.raises(ValueError, match=ENV_THRESHOLD):
        load_chat_service({**env, ENV_THRESHOLD: "high"})


def test_info_exposes_no_paths_or_faq_text(service):
    text = repr(service.info())
    assert "model_version" in text and "joblib" not in text and "FAQ-" not in text and "payroll@" not in text
