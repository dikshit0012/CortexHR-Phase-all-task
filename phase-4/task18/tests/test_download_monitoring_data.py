import hashlib
import io
import zipfile

import pytest

import download_monitoring_data as downloader


class FakeResponse(io.BytesIO):
    headers = {}

    def __init__(self, content, url="https://files.example.test/archive.zip"):
        super().__init__(content)
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self, content):
        self.content = content

    def open(self, url, timeout):
        return FakeResponse(self.content)


def archive_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def fetch_fixture(monkeypatch, content):
    monkeypatch.setattr(downloader.urllib.request, "build_opener", lambda *_: FakeOpener(content))
    return "https://files.example.test/archive.zip", hashlib.sha256(content).hexdigest()


def test_missing_secrets_name_the_required_settings_without_leaking_values(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(env_file))
    monkeypatch.delenv("DATA_URL", raising=False)
    monkeypatch.delenv("DATA_SHA256", raising=False)

    assert downloader.main(["--data-dir", str(tmp_path / "private-data")]) == 1
    output = capsys.readouterr().out
    assert "CORTEXHR_MONITORING_DATA_URL" in output and "CORTEXHR_MONITORING_DATA_SHA256" in output
    assert "https://" not in output
    assert "CORTEXHR_REFERENCE_WORKBOOK=" in env_file.read_text()
    assert f"CORTEXHR_SALARY_CSV={tmp_path / 'private-data' / 'salary_training.csv'}" in env_file.read_text()


def test_valid_zip_is_checksum_verified_and_extracted_safely(tmp_path, monkeypatch):
    files = {path: b"synthetic test content" for path in (
        "attrition_dataset.csv",
        "attrition_scoring_2026.csv",
        "salary_training.csv",
        "task11_data/job_descriptions.csv",
        "task11_data/resumes.csv",
        "resume_bias_pairs.csv",
        "private/unlock_CX-15_protected_attributes.csv",
    )}
    content = archive_bytes(files)
    url, checksum = fetch_fixture(monkeypatch, content)
    data_dir = tmp_path / "data"

    assert downloader.fetch_and_extract(url, checksum, data_dir) == ()
    for path, expected in files.items():
        assert (data_dir / path).read_bytes() == expected


def test_main_exports_existing_csv_paths_to_github_environment(tmp_path, monkeypatch, capsys):
    files = {"attrition_dataset.csv": b"reference", "salary_training.csv": b"salary"}
    content = archive_bytes(files)
    env_file = tmp_path / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(env_file))
    monkeypatch.setenv("DATA_URL", "https://files.example.test/archive.zip")
    monkeypatch.setenv("DATA_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(downloader.urllib.request, "build_opener", lambda *_: FakeOpener(content))

    assert downloader.main(["--data-dir", str(tmp_path / "data")]) == 0
    environment = env_file.read_text()
    assert f"CORTEXHR_REFERENCE_CSV={tmp_path / 'data' / 'attrition_dataset.csv'}" in environment
    assert f"CORTEXHR_SALARY_CSV={tmp_path / 'data' / 'salary_training.csv'}" in environment
    assert "NOT_EVALUATED" in capsys.readouterr().out


@pytest.mark.parametrize(
    "content,checksum,url,error",
    [
        (b"not a zip", hashlib.sha256(b"not a zip").hexdigest(), "https://files.example.test/a.zip", "ZIP structure or central directory"),
        (archive_bytes({}), hashlib.sha256(archive_bytes({})).hexdigest(), "https://files.example.test/a.zip", "ZIP is empty"),
        (archive_bytes({"resume_bias_pairs.csv": b""}), hashlib.sha256(archive_bytes({"resume_bias_pairs.csv": b""})).hexdigest(), "https://files.example.test/a.zip", "empty input file"),
        (archive_bytes({"../../outside.csv": b"no"}), None, "https://files.example.test/a.zip", "unsafe path"),
        (archive_bytes({"resume_bias_pairs.csv": b"ok"}), "0" * 64, "https://files.example.test/a.zip", "does not match"),
        (archive_bytes({"resume_bias_pairs.csv": b"ok"}), "bad-checksum", "https://files.example.test/a.zip", "exactly 64 hexadecimal"),
        (archive_bytes({"resume_bias_pairs.csv": b"ok"}), None, "http://files.example.test/a.zip", "direct HTTPS"),
    ],
)
def test_invalid_or_untrusted_archive_is_not_extracted(tmp_path, monkeypatch, content, checksum, url, error):
    if checksum is None and url.startswith("https:"):
        checksum = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(downloader.urllib.request, "build_opener", lambda *_: FakeOpener(content))
    data_dir = tmp_path / "data"

    with pytest.raises(downloader.DownloadError, match=error):
        downloader.fetch_and_extract(url, checksum, data_dir)
    assert not data_dir.exists()
    assert not (tmp_path / "outside.csv").exists()


def test_redirect_cannot_downgrade_from_https():
    request = downloader.urllib.request.Request("https://files.example.test/archive.zip")
    handler = downloader.HTTPSOnlyRedirectHandler()

    with pytest.raises(downloader.DownloadError, match="direct HTTPS"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://files.example.test/archive.zip")


def test_corrupt_zip_payload_is_not_extracted(tmp_path, monkeypatch):
    damaged = bytearray(archive_bytes({"resume_bias_pairs.csv": b"valid payload"}))
    name_length = int.from_bytes(damaged[26:28], "little")
    extra_length = int.from_bytes(damaged[28:30], "little")
    damaged[30 + name_length + extra_length] ^= 0xFF
    content = bytes(damaged)
    url, checksum = fetch_fixture(monkeypatch, content)
    data_dir = tmp_path / "data"

    with pytest.raises(downloader.DownloadError):
        downloader.fetch_and_extract(url, checksum, data_dir)
    assert not data_dir.exists()


def test_invalid_central_directory_explains_how_to_recover(tmp_path, monkeypatch):
    damaged = bytearray(archive_bytes({"resume_bias_pairs.csv": b"valid payload"}))
    central_directory = damaged.find(b"PK\x01\x02")
    assert central_directory >= 0
    damaged[central_directory:central_directory + 4] = b"FAIL"
    content = bytes(damaged)
    url, checksum = fetch_fixture(monkeypatch, content)

    with pytest.raises(downloader.DownloadError, match="regenerate the ZIP from the original files"):
        downloader.fetch_and_extract(url, checksum, tmp_path / "data")


def test_incomplete_zip_reports_missing_input_groups(tmp_path, monkeypatch):
    content = archive_bytes({"task11_data/job_descriptions.csv": b"jobs"})
    url, checksum = fetch_fixture(monkeypatch, content)

    missing = downloader.fetch_and_extract(url, checksum, tmp_path / "data")
    assert "Task 11 resumes" in missing
    assert "Task 15 protected labels" in missing
    assert "Task 11 job descriptions" not in missing
