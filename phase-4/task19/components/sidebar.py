from datetime import datetime

import streamlit as st

from api_client import ApiError
from components import cache

PAGES = [
    ("Overview", "Start here: system status and what each tool does"),
    ("Attrition Risks", "Estimate employees' 12-month attrition probability"),
    ("Anomalies", "Check attendance or payroll records for unusual patterns"),
    ("HR Assistant", "Ask HR policy questions in plain language"),
    ("Resume Screener", "Rank candidates against a job description"),
    ("Pay Bands", "Recommended pay for a role profile"),
    ("Workforce Forecast", "See the headcount outlook with an uncertainty range"),
    ("Monitoring", "Service health and model status"),
]


def go_to(page: str) -> None:
    """Button callback (safe to set the nav widget's state from a callback)."""
    st.session_state["page"] = page


def render(client) -> str:
    st.sidebar.markdown("## 🧠 CortexHR")
    st.sidebar.caption("Workforce Intelligence Console")
    page = st.sidebar.radio("Navigate", [p for p, _ in PAGES], key="page")
    st.sidebar.caption(dict(PAGES)[page])
    st.sidebar.divider()
    if st.sidebar.button("Check API connection"):
        cache.clear()
        try:
            client.health()
            st.session_state["api_status"] = ("ok", "Connected")
        except ApiError as e:
            st.session_state["api_status"] = ("down", e.user_message)
        st.session_state["last_checked"] = datetime.now().strftime("%d %b %Y %H:%M:%S")
    status = st.session_state.get("api_status")
    if status:
        st.sidebar.markdown(f"{'🟢' if status[0] == 'ok' else '🔴'} **API:** {status[1]}")
        st.sidebar.caption(f"Last checked: {st.session_state.get('last_checked', '—')}")
    else:
        st.sidebar.markdown("⚪ **API:** not checked yet")
    st.sidebar.caption("Fictional training data · Zenith Nova Technologies")
    return page
