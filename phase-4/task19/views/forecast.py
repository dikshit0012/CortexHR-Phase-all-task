import pandas as pd
import streamlit as st

from api_client import ApiError
from components import alerts, charts
from contracts import FORECAST_DEPARTMENTS


def _show(r: dict):
    mode = r["mode"]
    st.markdown(alerts.badge(mode.upper(), "#15803d" if mode == "live" else "#b45309") + f" &nbsp; trained through **{r['trained_through']}**",
                unsafe_allow_html=True)
    if mode == "precomputed":
        st.caption("Precomputed: a fixed result from the committed Task 16 run, not a live prediction.")
    st.plotly_chart(charts.forecast_chart(r["forecast"], r["department_id"]), width="stretch")
    st.caption(f"Shaded band = {r.get('interval', 0.8):.0%} prediction interval. The API returns the forecast only, not historical headcount, so no historical line is drawn.")
    df = pd.DataFrame(r["forecast"])
    st.dataframe(df.rename(columns={"month": "Month", "expected_headcount": "Expected headcount", "lower_80": "Lower bound (80%)", "upper_80": "Upper bound (80%)"}),
                 width="stretch", hide_index=True)
    ev = r.get("event_adjustments") or []
    with st.expander(f"Workforce events in the model's history ({len(ev)})"):
        st.write(r.get("event_adjustment_note", ""))
        if ev:
            st.dataframe(pd.DataFrame(ev), width="stretch", hide_index=True)
    with st.expander("Method and notes"):
        st.write(f"**Method:** {r.get('method')}  \n**Model:** {r.get('model_version')}")
        st.write(r.get("note", ""))
    alerts.review_label("forecast")


def render(client, cfg):
    st.header("Workforce Forecast")
    st.write("Expected month-end headcount for the next months, with an uncertainty range.")
    with st.form("fc_form"):
        c1, c2 = st.columns(2)
        dept = c1.selectbox("Department", FORECAST_DEPARTMENTS, help="ALL = whole company")
        horizon = c2.slider("Months ahead", 1, 6, 6)
        go = st.form_submit_button("Load forecast", type="primary")
    if go and alerts.single_flight("fc_busy"):
        try:
            with st.spinner("Loading forecast…"):
                st.session_state["fc_res"] = client.forecast(dept, horizon)
        except ApiError as e:
            st.session_state.pop("fc_res", None)
            alerts.show_error(e)
        finally:
            alerts.release("fc_busy")
    if "fc_res" in st.session_state:
        _show(st.session_state["fc_res"])
    elif not go:
        alerts.empty_state('Choose a department and press "Load forecast".')
