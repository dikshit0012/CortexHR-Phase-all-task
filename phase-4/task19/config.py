"""Console configuration. Everything comes from environment variables or Streamlit secrets; nothing production-specific
is hardcoded."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

DEFAULT_TIMEOUT = 30.0   # free hosting can take a while to wake up after idling
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ConfigError(Exception):
    """Required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    api_url: str
    timeout: float = DEFAULT_TIMEOUT
    api_key: str | None = field(default=None, repr=False)            # sent as X-API-Key; never printed or shown
    console_password: str | None = field(default=None, repr=False)   # optional gate in front of the console


def _from_secrets(name: str):
    try:
        import streamlit as st
        v = st.secrets.get(name)
        return str(v) if v is not None else None
    except Exception:   # no secrets file, or not running under Streamlit
        return None


def _get(name: str, env):
    v = env.get(name)
    if v is None or not str(v).strip():
        v = _from_secrets(name)
    return str(v).strip() if v and str(v).strip() else None


def validate_url(url: str) -> str:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ConfigError("CORTEXHR_API_URL must be a full URL such as https://your-api.example.com")
    if p.scheme == "http" and p.hostname not in LOCAL_HOSTS:
        raise ConfigError("CORTEXHR_API_URL must use HTTPS (plain HTTP is only allowed for localhost).")
    if p.query or p.fragment or p.username or p.path.rstrip("/") not in ("", "/v1"):
        raise ConfigError("CORTEXHR_API_URL must be the API root only (no path, credentials, query or fragment).")
    return url.rstrip("/").removesuffix("/v1")


def load_config(env=None) -> Config:
    env = os.environ if env is None else env
    raw = _get("CORTEXHR_API_URL", env)
    if not raw:
        raise ConfigError("CORTEXHR_API_URL is not set. Set it as an environment variable or in .streamlit/secrets.toml.")
    api_url = validate_url(raw)
    t = _get("CORTEXHR_TIMEOUT_SECONDS", env)
    try:
        timeout = float(t) if t else DEFAULT_TIMEOUT
        if not 1 <= timeout <= 120:
            raise ValueError
    except ValueError:
        raise ConfigError("CORTEXHR_TIMEOUT_SECONDS must be a number between 1 and 120.")
    return Config(api_url=api_url, timeout=timeout, api_key=_get("CORTEXHR_API_KEY", env),
                  console_password=_get("CORTEXHR_CONSOLE_PASSWORD", env))
