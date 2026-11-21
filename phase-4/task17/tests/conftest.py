import os
import warnings
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

warnings.filterwarnings("ignore")
KEY = "test-key"
HEADERS = {"X-API-Key": KEY}
HEADCOUNT_CSV = os.environ.get("CORTEXHR_HEADCOUNT_CSV")
WORKBOOK = os.environ.get("CORTEXHR_WORKBOOK")


def make_client(**overrides):
    from cortexhr_api.config import Settings
    from cortexhr_api.main import create_app
    base = dict(api_keys=(KEY,), rate_limit_per_minute=10_000)
    if HEADCOUNT_CSV:
        base["headcount_csv"] = Path(HEADCOUNT_CSV)
    if WORKBOOK:
        base["workbook_path"] = Path(WORKBOOK)
    base.update(overrides)
    return TestClient(create_app(Settings(**base)))


@pytest.fixture(scope="session")
def client():
    with make_client() as c:                     # real artifacts, loaded once for the whole session
        yield c


@pytest.fixture
def headers():
    return dict(HEADERS)
