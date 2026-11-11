import pandas as pd
import pytest

from cortexhr_task9.data import DataError, load_test
from cortexhr_task9.inspection import inspect_datasets
from cortexhr_task9.paths import DEFAULT_FAQ, TEST_CSV, TRAIN_CSV


def _write_test_csv(tmp_path, rows):
    path = tmp_path / "test.csv"
    pd.DataFrame(rows, columns=["test_id", "text", "expected_intent"]).to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    "rows,message",
    [
        ([{"test_id": "T1", "text": " ", "expected_intent": "greet"}], "empty messages"),
        ([{"test_id": "T1", "text": "hello", "expected_intent": " "}], "without an intent label"),
        ([{"test_id": "", "text": "hello", "expected_intent": "greet"}], "without an ID"),
        (
            [
                {"test_id": "T1", "text": "hello", "expected_intent": "greet"},
                {"test_id": "T1", "text": "goodbye", "expected_intent": "goodbye"},
            ],
            "duplicate IDs",
        ),
    ],
)
def test_load_test_rejects_incomplete_or_duplicate_rows(tmp_path, rows, message):
    with pytest.raises(DataError, match=message):
        load_test(_write_test_csv(tmp_path, rows))


def test_dataset_inspection_is_reproducible_and_reports_real_data():
    report = inspect_datasets(TRAIN_CSV, TEST_CSV, DEFAULT_FAQ)

    assert report["training"]["rows"] == 858
    assert report["training"]["unique_intents"] == 33
    assert set(report["training"]["examples_per_intent"].values()) == {26}
    assert report["training"]["exact_duplicate_message_groups"] == 7
    assert report["training"]["rows_in_exact_duplicate_groups"] == 14
    assert report["training"]["duplicate_groups_after_typo_correction"] == 30
    assert report["training"]["rows_in_groups_after_typo_correction"] == 60
    assert report["training"]["conflicting_duplicate_groups"] == []
    assert report["training_test_overlap"]["exact_after_cleaning_test_rows"] == 0
    assert report["training_test_overlap"]["same_base_phrase_after_filler_removal_test_rows"] == 0
    assert report["faq"]["rows"] == 30 and report["faq"]["has_intent_column"] is False
    assert report["review_sample"]["rows_reviewed"] == 25
    assert report == inspect_datasets(TRAIN_CSV, TEST_CSV, DEFAULT_FAQ)
