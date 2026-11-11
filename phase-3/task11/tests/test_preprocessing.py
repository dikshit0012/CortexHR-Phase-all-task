import math

import pandas as pd
import pytest

from cortexhr_task11.preprocessing import (EXP_NONE, EXP_ROLES, EXP_SUMMARY, extract_experience_years, normalize_text, scrub_identity, tfidf_text)


def test_lowercase_special_chars_and_whitespace():
    assert normalize_text("  Built   CI/CD  &  A/B tests!!!  (C++, C#) \n\n Go-to-Market ") == "built ci/cd a/b tests c++ c# go-to-market"   # a lone "&" is noise and is dropped


@pytest.mark.parametrize("bad", [None, float("nan"), pd.NA, 123, b"Python  SQL", ""])
def test_normalize_never_raises(bad):
    assert isinstance(normalize_text(bad), str)


def test_literal_backslash_n_is_whitespace():
    assert normalize_text("SUMMARY\\nPython") == "summary python"


def test_tfidf_text_drops_numbers_and_durations():
    out = tfidf_text("Trained models that improved accuracy by 28% (2.2 yrs) serving 87k predictions")
    assert "28" not in out and "87k" not in out and "yrs" not in out and "2.2" not in out
    assert "trained models" in out


def test_experience_from_explicit_statement():
    assert extract_experience_years("Motivated ml engineer with 4.8 years of experience.") == (4.8, EXP_SUMMARY)
    assert extract_experience_years("WITH 3 YEARS OF EXPERIENCE") == (3.0, EXP_SUMMARY)


def test_experience_fallback_sum_of_role_durations():
    assert extract_experience_years("Engineer, X (2.2 yrs)\nAnalyst, Y (1 yrs)") == (3.2, EXP_ROLES)


@pytest.mark.parametrize("text", ["Owned a roadmap spanning 69 quarters", "Deployed a model serving 87k predictions a day",
                                  "Improved accuracy by 28%", "I completed 7 projects", "", None, float("nan")])
def test_other_numbers_are_not_experience(text):
    assert extract_experience_years(text) == (None, EXP_NONE)


def test_implausible_experience_is_rejected():
    assert extract_experience_years("with 200 years of experience") == (None, EXP_NONE)


def test_scrub_identity_removes_name_email_city_tier_and_education():
    raw = "FATHIMA FERNANDES   EXPERIENCE\nCoimbatore | fathima.fernandes@mail-demo.exampleSales Development Rep (0.3 yrs)\nEDUCATION\nBBA, Tier 1 institution"
    out = scrub_identity(raw, ["Fathima Fernandes", "fathima.fernandes@mail-demo.example", "Coimbatore", "BBA", "Tier 1"]).lower()
    for token in ("fathima", "fernandes", "mail-demo", "coimbatore", "bba", "tier 1"):
        assert token not in out
    assert "sales development rep" in out


def test_scrub_ignores_missing_identity_values():
    assert "python" in scrub_identity("Python", [None, float("nan"), "", pd.NA]).lower()


def test_scrub_removes_explicit_age_and_gender_but_preserves_experience_and_skills():
    examples = [
        "Male; he/him; 25 years old; Born in 2002. Python",
        "Female; she/her; Age: 24; Date of birth: 01/02/2002. SQL",
        "Non-binary; they/them; 25-year-old; Gender: non-binary. Power BI",
    ]
    for text, skill in zip(examples, ("python", "sql", "power bi")):
        out = scrub_identity(text).lower()
        for token in ("male", "female", "non-binary", "he", "him", "she", "her", "they", "them",
                      "age", "24", "25", "born", "2002", "gender", "date of birth"):
            assert token not in out
        assert skill in out


def test_scrub_removes_tier_and_university_prestige_but_preserves_qualifications():
    text = (
        "EDUCATION: B.Tech, Tier-1 college, Stanford University; "
        "Tier 3 college; University of Example; prestigious university; IIT Bombay. "
        "SKILLS: Python, SQL. CERTIFICATIONS: PMP."
    )
    out = scrub_identity(text, ["Alex Morgan", "Tier 1"]).lower()
    for token in ("tier", "stanford", "university", "college", "prestigious", "iit", "bombay"):
        assert token not in out
    assert "b.tech" in out
    assert "python" in out and "sql" in out and "pmp" in out
