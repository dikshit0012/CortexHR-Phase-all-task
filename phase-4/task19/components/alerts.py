"""Human-review labels, friendly errors, badges and double-submit protection."""
import streamlit as st

from api_client import ApiError
from contracts import LIMITATIONS, REVIEW_LABEL

BAND_COLORS = {"low": "#15803d", "medium": "#b45309", "high": "#b91c1c"}
STATUS_COLORS = {"OK": "#15803d", "WATCH": "#b45309", "ALERT": "#b91c1c"}
CHECK_COLORS = {**STATUS_COLORS, "NOT_EVALUATED": "#64748b", "ERROR": "#7f1d1d"}
CHECK_LABELS = {"NOT_EVALUATED": "NOT EVALUATED", "ERROR": "ERROR"}
CHECK_HELP = {**{}, "NOT_EVALUATED": "This check could not be run (missing data). It is not a pass.",
              "ERROR": "This check failed to run. It is not a pass."}
STATUS_HELP = {"OK": "Models and fairness checks are within expected ranges.",
               "WATCH": "Some measures are drifting. Review soon; no action is required yet.",
               "ALERT": "Significant drift or a fairness gap was found. Escalate to the data science owner."}


def review_label(kind: str) -> None:
    """Mandatory label, rendered right next to the result (never only in the footer)."""
    st.markdown(f'<div class="ai-label"><b>{REVIEW_LABEL}</b><br><span>{LIMITATIONS[kind]}</span></div>', unsafe_allow_html=True)


def show_error(err: ApiError) -> None:
    (st.info if err.kind in ("not_found", "unavailable") else st.error)(err.user_message)
    if err.request_id:
        st.caption(f"Reference for support: {err.request_id}")


def band_color(band) -> str:
    b = (band or "").lower()
    return next((c for k, c in BAND_COLORS.items() if k in b), "#475569")


def check_badge(status: str) -> str:
    return badge(CHECK_LABELS.get(status, status), CHECK_COLORS.get(status, "#475569"))


def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{text}</span>'


def empty_state(message: str) -> None:
    st.markdown(f'<div class="empty">{message}</div>', unsafe_allow_html=True)


def single_flight(key: str) -> bool:
    """False if a submission with this key is already running."""
    if st.session_state.get(key):
        return False
    st.session_state[key] = True
    return True


def release(key: str) -> None:
    st.session_state[key] = False
