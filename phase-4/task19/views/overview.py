import streamlit as st

from api_client import ApiError
from components import alerts, cache
from components.sidebar import go_to
from views.monitoring import overall_badge

NA = "Data unavailable"
CAPS = [("Attrition Risks", "Estimated chance an employee leaves within 12 months."),
        ("Anomalies", "Unusual attendance or payroll records worth a second look."),
        ("HR Assistant", "Plain-language answers to HR policy questions."),
        ("Resume Screener", "Rank applicants against a job description (decision support only)."),
        ("Pay Bands", "Recommended pay for a role profile."),
        ("Workforce Forecast", "Expected headcount for the coming months."),
        ("Monitoring", "Whether the service and its models are healthy.")]


def _card(col, title, value, muted=False):
    col.markdown(f'<div class="card"><h4>{title}</h4><div class="{"na" if muted else "v"}">{value}</div></div>', unsafe_allow_html=True)


def render(client, cfg):
    st.title("CortexHR – Workforce Intelligence Console")
    st.markdown('<div class="start"><h3>Start here: Attrition Risks</h3><p>Most HR managers begin by checking who may be at risk of leaving. '
                "Then use Anomalies, the Forecast and Monitoring to see the full picture.</p></div>", unsafe_allow_html=True)
    st.button("Open Attrition Risks →", type="primary", on_click=go_to, args=("Attrition Risks",))
    if not cfg.console_password:
        st.warning("This console is not password-protected. Anyone with the link can use it. Set CORTEXHR_CONSOLE_PASSWORD before sharing it.")

    h = ready = None
    try:
        h = cache.cached("health", 30, client.health)
        ready = cache.cached("ready", 30, client.ready)
    except ApiError as e:
        alerts.show_error(e)
    mon = None
    try:
        mon = cache.cached("monitoring", 60, client.monitoring)
    except ApiError:
        pass
    next_hc = None
    if h and "headcount_forecast" in h["models_available"]:
        try:
            f = cache.cached("fc_all", 300, lambda: client.forecast("ALL", 1))
            next_hc = f["forecast"][0]
        except ApiError:
            pass

    r1 = st.columns(4)
    _card(r1[0], "API connection", alerts.badge("Connected", "#15803d") if h else alerts.badge("Unavailable", "#b91c1c"))
    _card(r1[1], "Models available", f"{len(h['models_available'])} of {len(h['models'])}" if h else NA, muted=not h)
    _card(r1[2], "Ready to serve", ("Yes" if ready.get("ready") else "No") if ready else NA, muted=not ready)
    _card(r1[3], "Model monitoring (drift/fairness)", overall_badge(mon) if mon else "Not available yet", muted=not mon)
    r2 = st.columns(4)
    _card(r2[0], "Employees scored", NA, True); _card(r2[1], "High-risk employees", NA, True)
    _card(r2[2], "Anomalies found", NA, True)
    _card(r2[3], f"Forecast headcount, {next_hc['month'][:7]}" if next_hc else "Forecast headcount (next month)",
          f"{next_hc['expected_headcount']:,.0f} <span style='font-size:.8rem;color:#64748b'>(80%: {next_hc['lower_80']:,.0f}–{next_hc['upper_80']:,.0f})</span>" if next_hc else NA,
          muted=not next_hc)
    if next_hc:
        st.caption("Forecast: AI-generated, for human review. It is an estimate, not a guaranteed outcome.")
    st.caption("Totals of employees scored, high-risk employees and anomalies are not shown because the API has no aggregate endpoint for them. Nothing here is estimated or invented.")

    st.subheader("What you can do here")
    for name, text in CAPS:
        st.markdown(f"**{name}**: {text}")
