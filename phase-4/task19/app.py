"""CortexHR Insights Console (Streamlit). Run: streamlit run app.py"""
import logging

import streamlit as st

st.set_page_config(page_title="CortexHR Insights Console", page_icon="🧠", layout="wide")

from api_client import ApiClient  # noqa: E402
from components import gate, sidebar, theme  # noqa: E402
from config import ConfigError, load_config  # noqa: E402
from views import anomalies, assistant, forecast, monitoring, overview, payband, resume_screener, risks  # noqa: E402

logging.basicConfig(level=logging.INFO)
theme.inject()
try:
    cfg = load_config()
except ConfigError as e:
    st.error(f"Console is not configured. {e}")
    st.stop()
if not gate.check(cfg.console_password):
    st.stop()

client = ApiClient(cfg)
PAGES = {"Overview": overview, "Attrition Risks": risks, "Anomalies": anomalies, "HR Assistant": assistant,
         "Resume Screener": resume_screener, "Pay Bands": payband, "Workforce Forecast": forecast, "Monitoring": monitoring}
PAGES[sidebar.render(client)].render(client, cfg)
st.markdown('<div class="footer">CortexHR · Aura Workforce · Fictional training data (Zenith Nova Technologies). '
            "AI outputs are decision support and always need human review.</div>", unsafe_allow_html=True)
