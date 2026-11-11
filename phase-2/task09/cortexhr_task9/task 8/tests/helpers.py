"""Shared assertions."""
import re

from app import _paths  # noqa: F401
from cortexhr_task7.official_output import MODEL_VERSIONS, OFFICIAL_COLUMNS
from cortexhr_task7.reason_codes import reason_code_dictionary

CODES = reason_code_dictionary().set_index("reason_code")
PRIORITY = CODES["priority"].to_dict()


def assert_official_row(row: dict, source: str, record_id: str, flagged: str | None = None):
    """The response row has exactly the official columns, in order, with valid types / values."""
    assert list(row.keys()) == OFFICIAL_COLUMNS
    assert re.fullmatch(r"AN-\d{4,}", row["result_id"])
    assert row["source_dataset"] == source
    assert row["record_id"] == record_id
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["period"])
    assert 0.0 <= row["anomaly_score"] <= 1.0
    assert row["flagged"] in ("Y", "N")
    assert row["model_version"] == MODEL_VERSIONS[source]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", row["scored_at"])
    if flagged:
        assert row["flagged"] == flagged
    if row["flagged"] == "Y":
        assert row["reason_code"] in CODES.index                      # a real Task 7 reason code
        assert CODES.at[row["reason_code"], "source"] in source        # ...belonging to this detector
        assert isinstance(row["reason_text"], str) and len(row["reason_text"]) >= 10
        assert row["money_at_risk"] is None or row["money_at_risk"] >= 0
    else:
        assert row["reason_code"] is None and row["reason_text"] is None and row["money_at_risk"] is None
