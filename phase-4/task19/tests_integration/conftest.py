import os
import socket
import subprocess
import time

import pytest
import requests

KEY = "integration-test-key-0123456789"
PUBKEY = "integration-publish-key-9876543210"   # test-only values


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _start(repo, py, extra_env):
    port = _free_port()
    env = {**os.environ, "CORTEXHR_ENV": "integration-test", **extra_env}
    env.pop("CORTEXHR_API_KEYS", None) if "CORTEXHR_API_KEYS" not in extra_env else None
    proc = subprocess.Popen([py, "-m", "uvicorn", "cortexhr_api.main:app", "--port", str(port)], cwd=repo, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if requests.get(base + "/v1/health", timeout=1).ok:
                return proc, base
        except requests.RequestException:
            time.sleep(0.5)
    proc.kill()
    pytest.fail("API did not start")


def _need_repo():
    repo, py = os.environ.get("CORTEXHR_API_REPO"), os.environ.get("CORTEXHR_API_PYTHON")
    if not (repo and py):
        pytest.skip("set CORTEXHR_API_REPO and CORTEXHR_API_PYTHON (or CORTEXHR_TEST_API_URL) to run live tests")
    return repo, py


@pytest.fixture(scope="session")
def mon_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("monitoring")


@pytest.fixture(scope="session")
def api(mon_dir):
    """(base_url, api_key) of the real API."""
    if os.environ.get("CORTEXHR_TEST_API_URL"):
        yield os.environ["CORTEXHR_TEST_API_URL"].rstrip("/"), os.environ.get("CORTEXHR_TEST_API_KEY", "")
        return
    repo, py = _need_repo()
    proc, base = _start(repo, py, {"CORTEXHR_API_KEYS": KEY, "CORTEXHR_RATE_LIMIT_PER_MINUTE": "100000",
                                   "CORTEXHR_MONITORING_DIR": str(mon_dir), "CORTEXHR_MONITORING_PUBLISH_KEYS": PUBKEY,
                                   "CORTEXHR_MONITORING_MAX_AGE_HOURS": "48"})
    yield base, KEY
    proc.terminate()


@pytest.fixture(scope="session")
def rate_limited_api():
    repo, py = _need_repo()
    proc, base = _start(repo, py, {"CORTEXHR_API_KEYS": KEY, "CORTEXHR_RATE_LIMIT_PER_MINUTE": "3"})
    yield base, KEY
    proc.terminate()


@pytest.fixture(scope="session")
def unconfigured_auth_api():
    repo, py = _need_repo()
    proc, base = _start(repo, py, {"CORTEXHR_API_KEYS": "", "CORTEXHR_RATE_LIMIT_PER_MINUTE": "100000"})
    yield base, ""
    proc.terminate()


@pytest.fixture
def mon(api, mon_dir):
    """Monitoring write-side access to the SAME server (only possible when this suite started it)."""
    if os.environ.get("CORTEXHR_TEST_API_URL"):
        pytest.skip("monitoring publish tests need the suite-started API")
    f = mon_dir / "latest_report.json"
    f.unlink(missing_ok=True)
    yield {"base": api[0], "key": api[1], "pub": PUBKEY, "file": f}
    f.unlink(missing_ok=True)
