"""Optional access gate. The console holds the API key server-side and shows no keys, but anyone who can open the console
can use it, so production deployments should set CORTEXHR_CONSOLE_PASSWORD (or put the app behind Streamlit/SSO login)."""
import hmac

import streamlit as st

MAX_ATTEMPTS = 5


def check(password: str | None) -> bool:
    """Return True when the visitor may continue. Renders the password form otherwise."""
    if not password:
        return True
    if st.session_state.get("console_ok"):
        return True
    st.title("CortexHR – Workforce Intelligence Console")
    if st.session_state.get("console_fail", 0) >= MAX_ATTEMPTS:
        st.error("Too many incorrect attempts. Please reload the page later.")
        return False
    with st.form("gate"):
        entered = st.text_input("Console password", type="password")
        go = st.form_submit_button("Sign in", type="primary")
    if go:
        if hmac.compare_digest(entered.encode(), password.encode()):
            st.session_state["console_ok"] = True
            st.rerun()
        st.session_state["console_fail"] = st.session_state.get("console_fail", 0) + 1
        if st.session_state["console_fail"] >= MAX_ATTEMPTS:
            st.error("Too many incorrect attempts. Please reload the page later.")
        else:
            st.error("Incorrect password.")
    return False
