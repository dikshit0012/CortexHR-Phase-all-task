import pandas as pd
import streamlit as st

from api_client import ApiError
from components import alerts, charts, uploads
from contracts import ATTRITION_FIELDS, ATTRITION_GROUPS, ATTRITION_SAMPLE


def _input(name: str, meta: dict, key: str):
    default = ATTRITION_SAMPLE.get(name)
    label = name.replace("_", " ").capitalize()
    if meta.get("type") == "string":
        return st.text_input(label, value=default or "", max_chars=64, key=key)
    lo, hi = float(meta["min"]), float(meta["max"])
    step = 0.01 if hi <= 1 else (0.1 if hi <= 10 else 1.0)
    return st.number_input(label, min_value=lo, max_value=hi, value=float(default if default is not None else lo), step=step,
                           key=key, help=f"Allowed range: {lo:g} to {hi:g}")


def _results_df(res: dict, refs: list[str]) -> pd.DataFrame:
    rows = [{"label": r.get("record_ref") or refs[i], "probability": r["attrition_probability_12m"], "risk_band": r.get("risk_band")}
            for i, r in enumerate(res["results"])]
    return pd.DataFrame(rows)


def _show(res: dict, df: pd.DataFrame, key: str):
    st.markdown(f"**Model:** {res['model_version']} · **Employees scored:** {res['n_records']}")
    if len(df) == 1:
        p = float(df.loc[0, "probability"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Estimated probability of leaving within 12 months:** {p:.1%}")
            band = df.loc[0, "risk_band"]
            st.markdown("**Risk band:** " + (alerts.badge(band, alerts.band_color(band)) if band else
                                              "not available for a single employee (bands need a batch of 10 or more)"),
                        unsafe_allow_html=True)
        c2.plotly_chart(charts.probability_gauge(p), width="stretch", key=f"{key}_g")
    else:
        d = df.sort_values("probability", ascending=False).reset_index(drop=True)
        view = d.assign(probability=d["probability"].map("{:.1%}".format)).rename(
            columns={"label": "Employee label", "probability": "Estimated probability", "risk_band": "Risk band"})
        if df["risk_band"].notna().any():
            st.plotly_chart(charts.band_counts(df), width="stretch", key=f"{key}_b")
        st.plotly_chart(charts.probability_hist(df), width="stretch", key=f"{key}_h")
        st.dataframe(view, width="stretch", hide_index=True)
        st.download_button("Download results (CSV)", d.to_csv(index=False).encode(), "attrition_results.csv", "text/csv", key=f"{key}_dl")
    if res.get("risk_band_note"):
        st.caption(res["risk_band_note"])
    st.caption("This is a **predicted** probability. It is not the employee's actual outcome and not a confirmed resignation. "
               "The API does not return reasons or explanations for a score.")
    alerts.review_label("risk")
    st.warning("Human review required: do not take employment decisions from this score alone.")


def render(client, cfg):
    st.header("Attrition Risks")
    st.write("Estimate how likely an employee is to leave within 12 months, using the saved attrition model. "
             "The model uses job, pay, career, performance and attendance facts, and never names or protected attributes.")
    one, batch = st.tabs(["One employee", "Batch (CSV)"])

    with one:
        with st.form("risk_form"):
            ref = st.text_input("Label for this result (optional; do not use a real name)", max_chars=64)
            vals = {}
            for i, (group, names) in enumerate(ATTRITION_GROUPS.items()):
                with st.expander(group, expanded=i == 0):
                    for n in names:
                        vals[n] = _input(n, ATTRITION_FIELDS[n], f"att_{n}")
            go = st.form_submit_button("Estimate attrition probability", type="primary")
        st.caption("Pre-filled values are an example profile from the API documentation. Replace them with real values.")
        if go and alerts.single_flight("risk_busy"):
            rec = {k: (v if v != "" else None) for k, v in vals.items()}
            if ref.strip():
                rec["record_ref"] = ref.strip()
            rec = {k: v for k, v in rec.items() if v is not None}
            try:
                with st.spinner("Scoring…"):
                    res = client.attrition_predict([rec])
                st.session_state["risk_one"] = (res, _results_df(res, ["Employee 1"]))
            except ApiError as e:
                st.session_state.pop("risk_one", None)
                alerts.show_error(e)
            finally:
                alerts.release("risk_busy")
        if "risk_one" in st.session_state:
            _show(*st.session_state["risk_one"], key="one")
        elif not go:
            alerts.empty_state("No employee scored yet. Check the values above and press the button.")

    with batch:
        st.write("Upload a CSV with one row per employee to score many at once. **Risk bands (High / Medium / Low) appear when the batch has at least 10 employees**, because the official bands are rank-based.")
        template = pd.DataFrame([{"record_ref": "row-1", **ATTRITION_SAMPLE}])
        st.download_button("Download CSV template", template.to_csv(index=False).encode(), "attrition_template.csv", "text/csv")
        st.caption(f"Columns must match the template. Up to {uploads.MAX_ROWS} rows. Leave a cell empty for a missing value. "
                   "Do not include names, e-mail addresses or other identifiers.")
        up = st.file_uploader("CSV file", type=["csv"], key="risk_csv")
        if up is not None and st.button("Score batch", type="primary") and alerts.single_flight("risk_busy"):
            try:
                df = uploads.read_csv_upload(up.name, up.getvalue())
                recs = uploads.table_to_records(df, ATTRITION_FIELDS, extra_ok={"record_ref"})
                with st.spinner("Scoring…"):
                    res = client.attrition_predict(recs)
                st.session_state["risk_batch"] = (res, _results_df(res, [f"Row {i}" for i in range(1, len(recs) + 1)]))
            except uploads.UploadError as e:
                st.session_state.pop("risk_batch", None)
                st.error(str(e))
            except ApiError as e:
                st.session_state.pop("risk_batch", None)
                alerts.show_error(e)
            finally:
                alerts.release("risk_busy")
        if "risk_batch" in st.session_state:
            _show(*st.session_state["risk_batch"], key="batch")
        elif up is None:
            alerts.empty_state("Upload a CSV to score a batch.")
