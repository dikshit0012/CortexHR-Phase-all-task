"""Reproducible, read-only inspection of the supplied Task 9 datasets."""
from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .data import DataError, phrase_group, sha256_file
from .text_cleaning import clean_text, load_typo_map

_TOKEN = re.compile(r"[a-z0-9&']+")
_INFORMAL = re.compile(r"\b(?:pls|yaar|hey|yo|thx|u|r)\b", re.IGNORECASE)


def _read_csv(path: Path, required: tuple[str, ...], dataset: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"{dataset} data file not found: {path.name}")
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataError(f"{dataset} data is missing columns {missing}")
    return frame


def _text_signals(text: str, typo_map: Dict[str, str]) -> list[str]:
    signals = []
    if any(char.isupper() for char in text):
        signals.append("uppercase")
    if re.search(r"\s{2,}", text):
        signals.append("repeated_whitespace")
    if re.search(r"([?!.,;:])\1+", text):
        signals.append("repeated_punctuation")
    tokens = _TOKEN.findall(text.lower())
    if any(token in typo_map for token in tokens):
        signals.append("known_typo")
    if _INFORMAL.search(text):
        signals.append("informal_wording")
    return signals


def inspect_datasets(
    training_path: Path,
    test_path: Path,
    faq_path: Path,
    sample_size: int = 25,
    seed: int = 42,
) -> Dict[str, Any]:
    """Return dataset quality, overlap, and a deterministic sample of real training rows."""
    if sample_size < 0:
        raise ValueError("sample_size must not be negative")

    train = _read_csv(training_path, ("example_id", "text", "intent"), "training")
    test = _read_csv(test_path, ("test_id", "text", "expected_intent"), "test")
    faq = _read_csv(faq_path, ("faq_id", "question", "answer", "category", "policy_source"), "FAQ")

    train_texts = train["text"].astype(str).tolist()
    test_texts = test["text"].astype(str).tolist()
    train_clean = [clean_text(text) for text in train_texts]
    test_clean = [clean_text(text) for text in test_texts]

    raw_duplicate_groups: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(train_texts):
        key = " ".join(text.lower().split())
        if key:
            raw_duplicate_groups[key].append(index)
    raw_duplicate_groups = {text: indexes for text, indexes in raw_duplicate_groups.items() if len(indexes) > 1}

    duplicate_groups: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(train_clean):
        if text:
            duplicate_groups[text].append(index)
    duplicate_groups = {text: indexes for text, indexes in duplicate_groups.items() if len(indexes) > 1}
    conflict_groups = [
        {
            "text": text,
            "intents": sorted({train.iloc[index]["intent"].strip() for index in indexes}),
            "row_count": len(indexes),
        }
        for text, indexes in duplicate_groups.items()
        if len({train.iloc[index]["intent"].strip() for index in indexes}) > 1
    ]

    intent_counts = train["intent"].str.strip().value_counts().sort_index()
    train_base = {phrase_group(text) for text in train_texts if phrase_group(text)}
    train_clean_set = set(train_clean)
    exact_overlap = sum(text in train_clean_set for text in test_clean if text)
    base_overlap = sum(phrase_group(text) in train_base for text in test_texts if phrase_group(text))

    typo_map = load_typo_map()
    sample_count = min(sample_size, len(train))
    sample_indexes = random.Random(seed).sample(range(len(train)), sample_count) if sample_count else []
    review_sample = []
    for index in sample_indexes:
        row = train.iloc[index]
        text = str(row["text"])
        review_sample.append(
            {
                "example_id": str(row["example_id"]),
                "text": text,
                "intent": str(row["intent"]).strip(),
                "signals": _text_signals(text, typo_map),
            }
        )

    class_counts = {str(intent): int(count) for intent, count in intent_counts.items()}
    min_class = min(class_counts.values(), default=0)
    max_class = max(class_counts.values(), default=0)
    return {
        "files": {
            "training": {"name": Path(training_path).name, "sha256": sha256_file(Path(training_path))},
            "test": {"name": Path(test_path).name, "sha256": sha256_file(Path(test_path))},
            "faq": {"name": Path(faq_path).name, "sha256": sha256_file(Path(faq_path))},
        },
        "training": {
            "rows": int(len(train)),
            "columns": list(train.columns),
            "unique_intents": len(class_counts),
            "examples_per_intent": class_counts,
            "empty_messages": int(train["text"].map(lambda text: not clean_text(text)).sum()),
            "empty_intent_labels": int(train["intent"].str.strip().eq("").sum()),
            "exact_duplicate_message_groups": len(raw_duplicate_groups),
            "rows_in_exact_duplicate_groups": sum(len(indexes) for indexes in raw_duplicate_groups.values()),
            "duplicate_groups_after_typo_correction": len(duplicate_groups),
            "rows_in_groups_after_typo_correction": sum(len(indexes) for indexes in duplicate_groups.values()),
            "conflicting_duplicate_groups": conflict_groups,
            "class_imbalance": {
                "minimum_examples": min_class,
                "maximum_examples": max_class,
                "max_to_min_ratio": round(max_class / min_class, 4) if min_class else None,
            },
            "text_signal_counts": {
                "uppercase": int(sum(any(c.isupper() for c in text) for text in train_texts)),
                "repeated_whitespace": int(sum(bool(re.search(r"\s{2,}", text)) for text in train_texts)),
                "repeated_punctuation": int(sum(bool(re.search(r"([?!.,;:])\1+", text)) for text in train_texts)),
                "known_typo": int(sum("known_typo" in _text_signals(text, typo_map) for text in train_texts)),
                "informal_wording": int(sum("informal_wording" in _text_signals(text, typo_map) for text in train_texts)),
            },
        },
        "test": {
            "rows": int(len(test)),
            "columns": list(test.columns),
            "empty_messages": int(test["text"].map(lambda text: not clean_text(text)).sum()),
            "empty_intent_labels": int(test["expected_intent"].str.strip().eq("").sum()),
        },
        "faq": {
            "rows": int(len(faq)),
            "columns": list(faq.columns),
            "has_intent_column": "intent" in faq.columns,
        },
        "training_test_overlap": {
            "exact_after_cleaning_test_rows": exact_overlap,
            "same_base_phrase_after_filler_removal_test_rows": base_overlap,
            "base_phrase_method": "Exact equality after clean_text and removal of configured filler words; semantic paraphrases are not detected.",
        },
        "review_sample": {
            "seed": seed,
            "rows_reviewed": sample_count,
            "messages": review_sample,
        },
    }
