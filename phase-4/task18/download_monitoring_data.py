#!/usr/bin/env python
"""Download and safely extract the private Task 18 input ZIP."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit

MAX_DOWNLOAD_BYTES = 2 * 1024**3
MAX_UNCOMPRESSED_BYTES = 4 * 1024**3
MAX_ZIP_ENTRIES = 64

ALLOWED_FILES = {
    "attrition_dataset.csv",
    "attrition_scoring_2026.csv",
    "salary_training.csv",
    "CortexHR_Dataset.xlsx",
    "02_CortexHR_AI_ML_Dataset.xlsx",
    "task11_data/job_descriptions.csv",
    "task11_data/resumes.csv",
    "task11_data/resume_job_labels.csv",
    "resume_bias_pairs.csv",
    "private/unlock_CX-15_protected_attributes.csv",
}
ALLOWED_DIRECTORIES = {"task11_data", "private"}
REQUIRED_INPUT_GROUPS = (
    ("attrition reference", ("attrition_dataset.csv", "02_CortexHR_AI_ML_Dataset.xlsx")),
    ("attrition scoring", ("attrition_scoring_2026.csv", "CortexHR_Dataset.xlsx")),
    ("salary training", ("salary_training.csv", "CortexHR_Dataset.xlsx")),
    ("Task 11 job descriptions", ("task11_data/job_descriptions.csv",)),
    ("Task 11 resumes", ("task11_data/resumes.csv",)),
    ("Task 12 twin pairs", ("resume_bias_pairs.csv",)),
    ("Task 15 protected labels", ("private/unlock_CX-15_protected_attributes.csv",)),
)


class DownloadError(Exception):
    """A safe, user-facing download or archive validation error."""


def _validate_https_url(url: str) -> None:
    if "\r" in url or "\n" in url:
        raise DownloadError("the configured URL is invalid")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
    except ValueError:
        raise DownloadError("the configured URL is invalid") from None
    if parsed.scheme.lower() != "https" or not hostname or parsed.username or parsed.password:
        raise DownloadError("the data URL must be a direct HTTPS URL without embedded credentials")


class HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_https_url(urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _write_github_environment(data_dir: Path) -> None:
    env_file = os.environ.get("GITHUB_ENV")
    if not env_file:
        return
    values = {
        "CORTEXHR_WORKBOOK": data_dir / "CortexHR_Dataset.xlsx",
        "CORTEXHR_REFERENCE_WORKBOOK": data_dir / "02_CortexHR_AI_ML_Dataset.xlsx",
        "CORTEXHR_SCORING_WORKBOOK": data_dir / "CortexHR_Dataset.xlsx",
        "CORTEXHR_TASK11_DATA": data_dir / "task11_data",
        "CORTEXHR_TASK12_PAIRS": data_dir / "resume_bias_pairs.csv",
        "CORTEXHR_TASK12_PRIVATE": data_dir / "private",
        "CORTEXHR_TASK15_PRIVATE": data_dir / "private",
        "CORTEXHR_SALARY_CSV": data_dir / "salary_training.csv",
    }
    for name, filename in (
        ("CORTEXHR_REFERENCE_CSV", "attrition_dataset.csv"),
        ("CORTEXHR_SCORING_CSV", "attrition_scoring_2026.csv"),
    ):
        if (data_dir / filename).is_file():
            values[name] = data_dir / filename
    with open(env_file, "a", encoding="utf-8") as stream:
        for name, value in values.items():
            stream.write(f"{name}={value}\n")


def _download_archive(url: str, expected_sha256: str, archive_path: Path) -> None:
    _validate_https_url(url)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise DownloadError("CORTEXHR_MONITORING_DATA_SHA256 must contain exactly 64 hexadecimal characters")

    opener = urllib.request.build_opener(HTTPSOnlyRedirectHandler())
    digest = hashlib.sha256()
    total = 0
    try:
        with opener.open(url, timeout=60) as response:
            _validate_https_url(response.geturl())
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > MAX_DOWNLOAD_BYTES:
                        raise DownloadError("the monitoring ZIP exceeds the allowed download size")
                except ValueError:
                    raise DownloadError("the download returned an invalid content length") from None
            with archive_path.open("xb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise DownloadError("the monitoring ZIP exceeds the allowed download size")
                    digest.update(chunk)
                    target.write(chunk)
    except DownloadError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise DownloadError(f"the HTTPS download failed ({type(exc).__name__})") from None

    if total == 0:
        raise DownloadError("the downloaded monitoring ZIP is empty")
    if digest.hexdigest().lower() != expected_sha256.lower():
        raise DownloadError("the downloaded ZIP SHA-256 does not match the configured checksum")


def _validate_archive(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = archive.infolist()
    if not infos or len(infos) > MAX_ZIP_ENTRIES:
        raise DownloadError("the ZIP is empty or contains too many entries")

    files = []
    seen = set()
    total_size = 0
    for info in infos:
        name = info.filename
        if "\\" in name or name.startswith("/"):
            raise DownloadError("the ZIP contains an unsafe path")
        clean_name = name.rstrip("/")
        path = PurePosixPath(clean_name)
        if not clean_name or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise DownloadError("the ZIP contains an unsafe path")
        if path.as_posix() != clean_name or re.match(r"^[A-Za-z]:", clean_name):
            raise DownloadError("the ZIP contains an unsafe path")

        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if stat.S_ISLNK(mode) or file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise DownloadError("the ZIP contains an unsupported file type")
        if info.flag_bits & 0x1:
            raise DownloadError("encrypted ZIP entries are not supported")

        if info.is_dir():
            if clean_name not in ALLOWED_DIRECTORIES:
                raise DownloadError("the ZIP contains an unsupported path")
            continue
        if clean_name not in ALLOWED_FILES or clean_name in seen:
            raise DownloadError("the ZIP contains an unsupported or duplicate path")
        if info.file_size == 0:
            raise DownloadError("the ZIP contains an empty input file")
        seen.add(clean_name)
        total_size += info.file_size
        if info.file_size > MAX_UNCOMPRESSED_BYTES or total_size > MAX_UNCOMPRESSED_BYTES:
            raise DownloadError("the ZIP expands beyond the allowed size")
        files.append(info)

    if not files:
        raise DownloadError("the ZIP contains no supported input files")
    return files


def _extract_archive(archive_path: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            files = _validate_archive(archive)
            if archive.testzip() is not None:
                raise DownloadError("the ZIP failed its CRC integrity check")
            for info in files:
                target = destination.joinpath(*PurePosixPath(info.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    except DownloadError:
        raise
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError, EOFError, zlib.error):
        raise DownloadError("the downloaded file is not a valid, readable ZIP archive") from None


def fetch_and_extract(url: str, expected_sha256: str, data_dir: Path) -> tuple[str, ...]:
    data_dir = data_dir.absolute()
    if data_dir.exists():
        raise DownloadError("refusing to reuse an existing monitoring data directory")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir.parent, prefix="cortexhr-download-") as work:
        work_dir = Path(work)
        archive_path = work_dir / "data.zip"
        staging = work_dir / "extracted"
        staging.mkdir()
        _download_archive(url, expected_sha256, archive_path)
        _extract_archive(archive_path, staging)
        os.replace(staging, data_dir)
    return tuple(
        label
        for label, alternatives in REQUIRED_INPUT_GROUPS
        if not any((data_dir / name).is_file() for name in alternatives)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_parent = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir()))
    parser.add_argument("--data-dir", type=Path, default=default_parent / "cortexhr-data")
    args = parser.parse_args(argv)
    data_dir = args.data_dir.absolute()

    if data_dir.exists():
        print("::error::Refusing to reuse an existing monitoring data directory.")
        return 1

    url = os.environ.get("DATA_URL", "")
    checksum = os.environ.get("DATA_SHA256", "")
    missing_secrets = [
        name
        for name, value in (
            ("CORTEXHR_MONITORING_DATA_URL", url),
            ("CORTEXHR_MONITORING_DATA_SHA256", checksum),
        )
        if not value
    ]
    if missing_secrets:
        _write_github_environment(data_dir)
        print(f"::error::Required repository secret(s) are missing: {', '.join(missing_secrets)}.")
        print("Add them under repository Settings > Secrets and variables > Actions.")
        return 1

    try:
        _validate_https_url(url)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::add-mask::{url}")
        missing_inputs = fetch_and_extract(url, checksum, data_dir)
    except DownloadError as exc:
        _write_github_environment(data_dir)
        print(f"::error::{exc}")
        return 1
    except OSError as exc:
        _write_github_environment(data_dir)
        print(f"::error::Monitoring data could not be staged ({type(exc).__name__}).")
        return 1

    _write_github_environment(data_dir)
    if missing_inputs:
        print("::warning::Verified ZIP is missing required input groups: "
              + ", ".join(missing_inputs)
              + "; affected checks will be NOT_EVALUATED.")
    else:
        print("Monitoring ZIP checksum, integrity, and required input layout verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
