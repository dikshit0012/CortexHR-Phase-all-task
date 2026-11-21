import pytest
from config import ConfigError, load_config


def test_missing_url():
    with pytest.raises(ConfigError):
        load_config({})


@pytest.mark.parametrize("bad", ["not a url", "ftp://x.com", "http://prod.example.com", "https://u:p@x.com", "https://x.com?a=1",
                                 "https://x.com/some/path"])
def test_invalid_urls(bad):
    with pytest.raises(ConfigError):
        load_config({"CORTEXHR_API_URL": bad})


def test_normalisation():
    assert load_config({"CORTEXHR_API_URL": "https://api.example.com/"}).api_url == "https://api.example.com"
    assert load_config({"CORTEXHR_API_URL": "https://api.example.com/v1"}).api_url == "https://api.example.com"
    assert load_config({"CORTEXHR_API_URL": "http://localhost:8000"}).api_url == "http://localhost:8000"


def test_timeout_default_and_validation():
    assert load_config({"CORTEXHR_API_URL": "https://a.com"}).timeout == 30.0
    for bad in ("abc", "0", "500"):
        with pytest.raises(ConfigError):
            load_config({"CORTEXHR_API_URL": "https://a.com", "CORTEXHR_TIMEOUT_SECONDS": bad})


def test_secrets_never_in_repr():
    c = load_config({"CORTEXHR_API_URL": "https://a.com", "CORTEXHR_API_KEY": "k-secret", "CORTEXHR_CONSOLE_PASSWORD": "pw-secret"})
    assert "k-secret" not in repr(c) and "pw-secret" not in repr(c) and c.api_key == "k-secret"
