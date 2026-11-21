import pytest
from components import uploads as u
from contracts import ATTRITION_FIELDS, ATTRITION_SAMPLE, ANOMALY_FIELDS, ATTENDANCE_SAMPLE
import pandas as pd


def csv_bytes(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode()


def test_csv_ok_and_types():
    df = u.read_csv_upload("a.csv", csv_bytes([{"record_ref": "r1", **ATTRITION_SAMPLE}]))
    rec = u.table_to_records(df, ATTRITION_FIELDS, extra_ok={"record_ref"})[0]
    assert rec["job_level"] == 3.0 and rec["department_id"] == "D01" and rec["record_ref"] == "r1"


def test_blank_cell_becomes_missing():
    row = {**ATTRITION_SAMPLE, "tenure_months": ""}
    rec = u.table_to_records(u.read_csv_upload("a.csv", csv_bytes([row])), ATTRITION_FIELDS)[0]
    assert "tenure_months" not in rec


@pytest.mark.parametrize("name,data,msg",
                         [pytest.param("a.xlsx", b"x", ".csv", id="wrong-extension"),
                          pytest.param("a.csv", b"x" * 800_000, "too large", id="oversized"),
                          pytest.param("a.csv", b"", "no rows", id="empty"),
                          pytest.param("a.csv", b"a,b\n", "no rows", id="header-only")])
def test_csv_file_rejections(name, data, msg):
    with pytest.raises(u.UploadError, match=msg):
        u.read_csv_upload(name, data)


def test_too_many_rows():
    with pytest.raises(u.UploadError, match="limit"):
        u.read_csv_upload("a.csv", csv_bytes([{"tenure_months": 1}] * 101))


def test_identity_unknown_and_missing_columns():
    with pytest.raises(u.UploadError, match="personal identifiers"):
        u.table_to_records(pd.DataFrame([{**ATTRITION_SAMPLE, "name": "A B"}]).astype(str), ATTRITION_FIELDS)
    with pytest.raises(u.UploadError, match="Unknown column"):
        u.table_to_records(pd.DataFrame([{"tenure_months": "1", "bogus": "2"}]), ATTRITION_FIELDS)
    with pytest.raises(u.UploadError, match="Missing required"):
        u.table_to_records(pd.DataFrame([{"employee_id": "E1"}]), ANOMALY_FIELDS["attendance"])


@pytest.mark.parametrize("col,val,msg", [("tenure_months", "abc", "number"), ("tenure_months", "9999", "between"), ("job_level", "-1", "between"),
                                         ("attendance_rate_3m", "1.5", "between")])
def test_value_validation(col, val, msg):
    with pytest.raises(u.UploadError, match=msg):
        u.table_to_records(pd.DataFrame([{col: val}]), ATTRITION_FIELDS)


def test_integer_fields_reject_fractions():
    row = {k: str(v) for k, v in ATTENDANCE_SAMPLE.items()}; row["present_days"] = "3.5"
    with pytest.raises(u.UploadError, match="whole number"):
        u.table_to_records(pd.DataFrame([row]), ANOMALY_FIELDS["attendance"])


def test_resume_txt_rules():
    assert u.read_resume_txt("a.txt", b"hello") == "hello"
    for name, data, msg in [("a.pdf", b"x", "only .txt"), ("a.docx", b"x", "only .txt"), ("a.txt", b"", "empty"),
                            ("a.txt", b"\xff\xfe\x00", "UTF-8"), ("a.txt", b"x" * 50_000, "too large"), ("a.txt", b"a" * 30_000, "20,000 characters")]:
        with pytest.raises(u.UploadError, match=msg):
            u.read_resume_txt(name, data)
    with pytest.raises(u.UploadError, match="20,000"):
        u.read_resume_txt("a.txt", ("a" * 20_001).encode())
