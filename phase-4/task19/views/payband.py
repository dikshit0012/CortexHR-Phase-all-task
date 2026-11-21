import pandas as pd
import streamlit as st

from api_client import ApiError
from components import alerts
from contracts import PAYBAND_FIELDS, PAYBAND_SAMPLE


def render(client, cfg):
    st.header("Pay Bands")
    st.write("Get a model-recommended pay level for a **role profile** (not for a named person). Use it as one input when reviewing offers.")
    with st.form("payband_form"):
        vals = {}
        cols = st.columns(2)
        for i, (n, m) in enumerate(PAYBAND_FIELDS.items()):
            label = n.replace("_", " ").capitalize() + (" *" if m.get("required") else "")
            d = PAYBAND_SAMPLE.get(n)
            with cols[i % 2]:
                if m["type"] == "string":
                    vals[n] = st.text_input(label, value=d or "", max_chars=64, key=f"pb_{n}")
                elif m["type"] == "integer":
                    vals[n] = int(st.number_input(label, int(m["min"]), int(m["max"]), int(d), step=1, key=f"pb_{n}"))
                else:
                    vals[n] = float(st.number_input(label, float(m["min"]), float(m["max"]), float(d), step=0.1, key=f"pb_{n}"))
        go = st.form_submit_button("Get pay recommendation", type="primary")
    st.caption("Pre-filled values are an example from the API documentation.")
    if go and alerts.single_flight("pb_busy"):
        body = {k: v for k, v in vals.items() if v != ""}
        try:
            with st.spinner("Calculating…"):
                st.session_state["pb_res"] = client.payband(body)
        except ApiError as e:
            st.session_state.pop("pb_res", None)
            alerts.show_error(e)
        finally:
            alerts.release("pb_busy")
    r = st.session_state.get("pb_res")
    if r:
        st.metric("Recommended midpoint", f"{r['predicted_salary_mid']:,}")
        lo, hi = r.get("recommended_salary_low"), r.get("recommended_salary_high")
        if lo is not None and hi is not None:
            st.write(f"**Recommended band:** {lo:,} to {hi:,}")
        st.caption(f"Band status: {r['band_status']}")
        st.write("**Comparison with simpler models**")
        st.dataframe(pd.DataFrame({"Model": [k.replace("_", " ").capitalize() for k in r["comparison"]],
                                   "Estimate": [f"{v:,}" if v is not None else "n/a" for v in r["comparison"].values()]}),
                     width="stretch", hide_index=True)
        with st.expander("Fairness and decision notes"):
            st.write(r.get("fairness_note", "")); st.write(r.get("decision_note", ""))
        st.caption(f"Model: {r.get('model_version')} ({r.get('selected_model')}). Currency units are those of the training data.")
        alerts.review_label("payband")
    elif not go:
        alerts.empty_state("No recommendation yet.")
