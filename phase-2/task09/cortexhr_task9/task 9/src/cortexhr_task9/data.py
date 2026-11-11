"""Dataset loading and validation (columns are checked, never assumed)."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List

import pandas as pd

from .text_cleaning import clean_text

TRAIN_COLUMNS = ("example_id", "text", "intent")
TEST_COLUMNS = ("test_id", "text", "expected_intent")
_FILLER = re.compile(r"\b(please|pls|urgently|quick question|i want to know|can you tell me|hey|hi)\b")


class DataError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path, required) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"data file not found: {path.name}")
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{path.name} is missing required columns {missing}; found {list(df.columns)}")
    return df


def load_training(path: Path) -> pd.DataFrame:
    df = _read(path, TRAIN_COLUMNS)
    _validate_rows(df, "example_id", "intent", "training")
    df["intent"] = df["intent"].str.strip()
    return df


def load_test(path: Path) -> pd.DataFrame:
    df = _read(path, TEST_COLUMNS)
    _validate_rows(df, "test_id", "expected_intent", "test")
    df["expected_intent"] = df["expected_intent"].str.strip()
    return df


def _validate_rows(df: pd.DataFrame, id_column: str, label_column: str, dataset: str) -> None:
    ids = df[id_column].str.strip()
    if (ids == "").any():
        raise DataError(f"{dataset} data has rows without an ID")
    if ids.duplicated().any():
        raise DataError(f"{dataset} data has duplicate IDs")
    if (df["text"].map(clean_text) == "").any():
        raise DataError(f"{dataset} data has empty messages")
    if (df[label_column].str.strip() == "").any():
        raise DataError(f"{dataset} data has rows without an intent label")


def phrase_group(text: str) -> str:
    """The message with filler words and punctuation removed (typos already fixed by clean_text). Messages that differ only
    by filler ('pls', 'urgently', 'I want to know ...') share a group, so cross-validation never puts them on both sides."""
    s = re.sub(r"[?!.,:;]", " ", clean_text(text))
    return re.sub(r"\s+", " ", _FILLER.sub(" ", s)).strip()


def phrase_groups(texts) -> List[str]:
    return [phrase_group(t) for t in texts]
