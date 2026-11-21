import pandas as pd
import streamlit as st

from api_client import ApiError
from components import alerts, charts

CATEGORY = {"drift": "Input drift", "fairness": "Fairness", "missing_evidence": "Missing evidence",
            "execution_failure": "Monitoring failure", "data_integrity": "Data integrity"}
FAIR_ROWS = (("twin_resume", "Twin-resume fairness (Task 12)", "Identical resumes that differ only in a protected detail should score the same."),
             ("pay_gap", "Pay-gap fairness (Task 15)", "Pay recommendations should not drift apart between groups compared with the baseline."))


def overall_badge(rep: dict) -> str:
    """Badge text for the overall status. A stale report is never shown as simply healthy."""
    text = rep["status"] + (" · outdated" if rep["stale"] else "")
    return alerts.badge(text, alerts.STATUS_COLORS[rep["status"]])


def _health_section(client):
    try:
        h, rd = client.health(), client.ready()
    except ApiError as e:
        alerts.show_error(e)
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Application", h["application"].upper())
    c2.metric("Models loaded", f"{len(h['models_available'])} of {len(h['models'])}")
    c3.metric("Ready to serve", "Yes" if rd.get("ready") else "No")
    if not rd.get("ready"):
        st.warning("Not ready. Blocking: " + ", ".join(rd.get("blocking", [])))
    if rd.get("degraded"):
        st.info("Running with reduced capability. Unavailable: " + ", ".join(rd["degraded"]))
    st.dataframe(pd.DataFrame([{"Capability": n.replace("_", " ").capitalize(), "Status": m["status"], "Version": m.get("version") or "—",
                                "Mode": m.get("mode") or "—"} for n, m in h["models"].items()]), width="stretch", hide_index=True)
    st.caption(f"API version {h.get('api_version')} · environment: {h.get('environment')} · uptime {h.get('uptime_seconds', 0):,.0f}s")


def _report_section(rep: dict):
    st.markdown(overall_badge(rep), unsafe_allow_html=True)
    st.caption(alerts.STATUS_HELP[rep["status"]])
    if not rep["execution_ok"]:
        st.error("The monitoring run itself failed, so this status is not reliable. Ask the data science owner to investigate.")
    if not rep["evidence_complete"]:
        st.warning("Some checks could not be evaluated because evidence (data) was missing. A check that was not run is not a pass.")
    if rep["stale"]:
        st.warning(f"This report is {rep['report_age_hours']:.0f} hours old, which is older than the allowed age. "
                   "Treat the status as outdated until monitoring runs again.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Last completed run (UTC)", rep["last_run"].replace("T", " ").replace("Z", ""))
    c2.metric("Report age", f"{rep['report_age_hours']:.1f} h")
    c3.metric("Run outcome", rep["execution_status"].title())

    st.subheader("Input drift (PSI)")
    d = rep["drift"]
    st.plotly_chart(charts.drift_counts_chart(d), width="stretch", key="mon_drift")
    drifting = [r for r in rep["reasons"] if r["category"] == "drift"]
    if drifting:
        st.write("**Features with drift:**")
        st.dataframe(pd.DataFrame([{"Feature": r["item"], "Status": r["status"]} for r in drifting]), width="stretch", hide_index=True)
    elif d["alert"] == d["watch"] == d["error"] == d["not_evaluated"] == 0:
        st.success("No feature shows drift.")
    st.caption("PSI < 0.10 OK · 0.10–0.25 WATCH · > 0.25 ALERT. PSI measures a change in input data, not model accuracy. "
               "The API returns drift counts and feature names only; the per-feature PSI values are in the monitoring report itself.")

    st.subheader("Fairness checks")
    for key, title, what in FAIR_ROWS:
        status = rep["fairness"][key]["status"]
        st.markdown(f"**{title}** &nbsp; {alerts.check_badge(status)}", unsafe_allow_html=True)
        st.caption(what + " " + (alerts.CHECK_HELP.get(status) or alerts.STATUS_HELP.get(status, "")))
    st.caption("Fairness gaps are monitoring signals, not proof of discrimination.")

    other = [r for r in rep["reasons"] if r["category"] != "drift"]
    if other:
        st.subheader("Why the status is not OK")
        st.dataframe(pd.DataFrame([{"Category": CATEGORY.get(r["category"], r["category"].replace("_", " ").title()), "Check": r["check"].replace("_", " "),
                                    "Item": r["item"], "Status": r["status"]} for r in other]), width="stretch", hide_index=True)
    alerts.review_label("monitoring")


def render(client, cfg):
    st.header("Monitoring")
    st.write("Shows the latest completed drift and fairness report and the service's health. "
             "The console only reads the report: it never runs monitoring and cannot publish or change it.")
    st.subheader("Drift and fairness report")
    try:
        rep = client.monitoring()
    except ApiError as e:
        if e.kind == "not_found":
            st.info("No monitoring report has been published to the server yet, so no status is shown. "
                    "It appears here after the scheduled monitoring job has run and published its result.")
        else:
            alerts.show_error(e)
    else:
        _report_section(rep)
    st.subheader("Service health")
    _health_section(client)
