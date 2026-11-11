import math

import pytest

from cortexhr_task9.text_cleaning import clean_text, clean_texts, load_typo_map


@pytest.mark.parametrize("raw,expected", [
    ("  How do I apply for LEAVE  ", "how do i apply for leave"),
    ("is   my\tleave \n approved", "is my leave approved"),
    ("what??? is my salary!!", "what? is my salary!"),
    ("Where to get Form 16", "where to get form 16"),
    ("show Priya Nair\u2019s payslip", "show priya nair's payslip"),
    ("Data & AI team", "data & ai team"),
])
def test_basic_normalisation(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "\n\t", float("nan")])
def test_empty_and_missing_input_gives_empty_string(raw):
    assert clean_text(raw) == ""


def test_non_string_input_does_not_crash():
    assert clean_text(12345) == "12345"
    assert clean_text(b"Hello  BOT") == "hello bot"


@pytest.mark.parametrize("raw,expected", [
    ("is my laeve approved", "is my leave approved"),
    ("plesae what is my slaary date", "please what is my salary date"),
    ("hey thnaks", "hey thanks"),
])
def test_targeted_typo_corrections_from_the_data(raw, expected):
    assert clean_text(raw) == expected


def test_meaningful_tokens_are_not_removed_or_rewritten():
    assert clean_text("I do NOT want leave on 15 March") == "i do not want leave on 15 march"
    assert "not" in clean_text("not approved").split()
    assert clean_text("form 16") == "form 16"                 # 'form' must never become 'from'
    assert clean_text("pls urgently") == "pls urgently"       # filler is kept: the model learns its weight


def test_different_hr_questions_stay_different():
    assert clean_text("how many casual leaves") != clean_text("how many sick leaves")
    assert clean_text("leave balance") != clean_text("leave policy")


def test_cleaning_is_idempotent_and_deterministic():
    once = clean_text("  Plesae  show my   SLAAry??? ")
    assert clean_text(once) == once
    assert clean_texts(["A  b", None]) == ["a b", ""]


def test_typo_dictionary_is_safe():
    typos = load_typo_map()
    assert len(typos) > 50
    assert "form" not in typos and "may" not in typos                      # real words must not be rewritten
    assert all(k != v and k.isascii() for k, v in typos.items())
    assert not set(typos) & set(typos.values())                            # no chains, so one pass is enough
