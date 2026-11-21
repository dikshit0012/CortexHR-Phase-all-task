"""Tiny per-session TTL cache so reruns do not hammer the API (it rate-limits per client)."""
import time

import streamlit as st


def cached(key: str, ttl: float, fn):
    store = st.session_state.setdefault("_cache", {})
    hit = store.get(key)
    if hit and time.monotonic() - hit[0] < ttl:
        if isinstance(hit[1], Exception):
            raise hit[1]
        return hit[1]
    try:
        val = fn()
    except Exception as e:      # cache failures briefly too, so a down API is not retried on every click
        store[key] = (time.monotonic(), e)
        raise
    store[key] = (time.monotonic(), val)
    return val


def clear():
    st.session_state.pop("_cache", None)
