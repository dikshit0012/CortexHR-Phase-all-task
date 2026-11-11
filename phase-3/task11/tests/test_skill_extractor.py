import pandas as pd

from cortexhr_task11.skill_extractor import SkillExtractor, build_default_extractor, split_skill_list


def ex(*vocab):
    return SkillExtractor(list(vocab))


def test_word_boundaries():
    e = ex("SQL", "PostgreSQL", "Java", "JavaScript")
    assert e.extract("uses postgresql daily") == ["PostgreSQL"]
    assert e.extract("javascript only") == ["JavaScript"]
    assert set(e.extract("sql, java")) == {"SQL", "Java"}


def test_synonyms_share_one_key():
    e = ex("Excel", "MS Excel", "scikit-learn")
    assert e.canonical_key("MS Excel") == e.canonical_key("excel")
    assert e.extract("advanced MS Excel") == ["Excel"]
    assert e.extract("trained with sklearn") == ["scikit-learn"]


def test_all_caps_and_punctuation():
    assert "CI/CD" in ex("CI/CD").extract("INTRODUCED CI/CD, DOCKER")


def test_extracts_brief_examples():
    e = build_default_extractor()
    found = set(e.extract("Python Java SQL Machine Learning Deep Learning NLP TensorFlow PyTorch React Angular HTML CSS JavaScript AWS Docker Git"))
    assert {"Python", "Java", "SQL", "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch", "React", "Angular",
            "HTML", "CSS", "JavaScript", "AWS", "Docker", "Git"} <= found


def test_add_skills():
    e = ex("Python")
    e.add_skills(["Rust"])
    assert e.extract("rust and python") == ["Rust", "Python"] or set(e.extract("rust and python")) == {"Rust", "Python"}


def test_split_skill_list_is_safe():
    assert split_skill_list("Python; SQL ;Git;;python") == ["Python", "SQL", "Git"]
    assert split_skill_list("A, B") == ["A", "B"]
    for bad in (None, float("nan"), pd.NA, "", "   "):
        assert split_skill_list(bad) == []


def test_evidence_line():
    e = ex("Tally")
    assert "Tally Prime" in e.evidence("EDUCATION\nCERTIFICATIONS\nTally Prime Certified", "tally")
