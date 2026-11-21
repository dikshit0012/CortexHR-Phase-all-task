"""Schema-agnostic renderer for responses whose contract is not yet confirmed."""
import pandas as pd
import streamlit as st


def render_payload(data) -> None:
    if isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
        st.dataframe(pd.json_normalize(data), width="stretch", hide_index=True)
    elif isinstance(data, dict):
        scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
        if scalars:
            st.dataframe(pd.DataFrame({"Field": list(scalars), "Value": [str(v) for v in scalars.values()]}),
                         width="stretch", hide_index=True)
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                with st.expander(str(k).replace("_", " ").title(), expanded=True):
                    render_payload(v)
    else:
        st.write(data)
