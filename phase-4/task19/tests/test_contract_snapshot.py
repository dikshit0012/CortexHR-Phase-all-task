"""The forms must only use fields that exist in the snapshot taken from the live API."""
from contracts import ATTRITION_FIELDS, ATTRITION_GROUPS, ANOMALY_FIELDS, ANOMALY_SAMPLES, PAYBAND_FIELDS, PAYBAND_SAMPLE, ATTRITION_SAMPLE, SNAPSHOT


def test_attrition_groups_cover_every_field_exactly_once():
    grouped = [n for g in ATTRITION_GROUPS.values() for n in g]
    assert sorted(grouped) == sorted(ATTRITION_FIELDS) and len(grouped) == len(set(grouped)) == 33


def test_samples_only_use_known_fields_and_respect_bounds():
    assert set(ATTRITION_SAMPLE) <= set(ATTRITION_FIELDS)
    for k, v in ATTRITION_SAMPLE.items():
        m = ATTRITION_FIELDS[k]
        if m["type"] == "number":
            assert m["min"] <= v <= m["max"], k
    for kind, s in ANOMALY_SAMPLES.items():
        assert set(s) <= set(ANOMALY_FIELDS[kind]), kind
    assert set(PAYBAND_SAMPLE) <= set(PAYBAND_FIELDS)


def test_snapshot_has_expected_routes():
    assert "/v1/attrition/predict" in SNAPSHOT["paths"] and "/v1/monitoring" in SNAPSHOT["paths"]
