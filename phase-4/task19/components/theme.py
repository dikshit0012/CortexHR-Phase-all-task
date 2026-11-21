import streamlit as st

CSS = """
<style>
[data-testid="stSidebar"]{background:#0f1b3d}
[data-testid="stSidebar"] *{color:#e8ecf7}
[data-testid="stSidebar"] .stButton button{background:#2563eb;color:#fff;border:0}
.block-container{padding-top:2rem;max-width:1100px}
.ai-label{border-left:4px solid #2563eb;background:#eff4ff;color:#0f1b3d;padding:.6rem .9rem;border-radius:8px;margin:.8rem 0;font-size:.92rem}
.ai-label span{color:#334155}
.card{border:1px solid #e3e6ef;border-radius:12px;padding:.9rem 1rem;background:#fff;height:100%}
.card h4{margin:0 0 .2rem 0;font-size:.85rem;color:#5b6478;font-weight:600}
.card .v{font-size:1.4rem;font-weight:700;color:#0f1b3d}
.card .na{font-size:1rem;color:#64748b}
.badge{color:#fff;border-radius:99px;padding:2px 12px;font-weight:700;font-size:.85rem}
.empty{border:1px dashed #cbd5e1;border-radius:12px;padding:1.2rem;text-align:center;color:#64748b}
.start{background:linear-gradient(135deg,#0f1b3d,#1d3a7a);color:#fff;border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1rem}
.start h3,.start p{color:#fff;margin:.1rem 0}
.footer{color:#64748b;font-size:.8rem;text-align:center;margin-top:2.5rem;border-top:1px solid #e3e6ef;padding-top:.8rem}
@media(max-width:640px){.block-container{padding:1rem .8rem}.card .v{font-size:1.1rem}}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)
