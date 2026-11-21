import datetime as dt

import pandas as pd
import streamlit as st

from api_client import ApiError
from components import alerts, charts, uploads
from contracts import ANOMALY_FIELDS, ANOMALY_SAMPLES

TITLES = {"attendance": "Attendance", "payroll": "Payroll"}


def _widget(kind: str, name: str, meta: dict, default):
    key, label = f"an_{kind}_{name}", name.replace("_", " ").capitalize()
    if meta.get("enum"):
        opts = list(meta["enum"])
        return st.selectbox(label, opts, index=opts.index(default) if default in opts else 0, key=key)
    if meta.get("format") == "date":
        return st.date_input(label, value=dt.date.fromisoformat(default) if default else dt.date.today(), key=key).isoformat()
    if meta["type"] == "integer":
        return int(st.number_input(label, value=int(default) if default is not None else 0, step=1, key=key))
    if meta["type"] == "number":
        return float(st.number_input(label, value=float(default) if default is not None else 0.0, key=key))
    return st.text_input(label, value=default or "", max_chars=64, key=key)


def _results_df(res: dict) -> pd.DataFrame:
    return pd.DataFrame(res["results"])


def _show(kind: str, res: dict):
    df = _results_df(res)
    c1, c2, c3 = st.columns(3)
    c1.metric("Records checked", res["n_records"])
    c2.metric("Flagged for review", res["n_flagged"])
    mar = df.loc[df["flagged"] == "Y", "money_at_risk"].dropna() if "money_at_risk" in df else pd.Series(dtype=float)
    c3.metric("Money at risk (sum of flagged, as returned)", f"{mar.sum():,.2f}" if len(mar) else "n/a")
    cols = [c for c in ("result_id", "employee_id", "period", "anomaly_score", "flagged", "reason_code", "reason_text", "money_at_risk") if c in df]
    st.dataframe(df[cols].rename(columns=lambda c: c.replace("_", " ").capitalize()), width="stretch", hide_index=True)
    if len(df) > 1:
        a, b = st.columns(2)
        a.plotly_chart(charts.flagged_counts(df), width="stretch", key=f"{kind}_fc")
        b.plotly_chart(charts.score_distribution(df), width="stretch", key=f"{kind}_sd")
        if (df["flagged"] == "Y").any():
            st.plotly_chart(charts.reason_counts(df), width="stretch", key=f"{kind}_rc")
    for r in df[df["flagged"] == "Y"].to_dict("records"):
        with st.expander(f"{r['result_id']} · {r['employee_id']} · {r.get('reason_code') or 'no reason code'}", expanded=len(df) == 1):
            st.write(r.get("reason_text") or "No explanation text was returned.")
    st.caption(f"{res.get('note', '')}  Model: {res['model_version']}")
    alerts.review_label("anomaly")


def _kind(client, kind: str):
    fields, sample = ANOMALY_FIELDS[kind], ANOMALY_SAMPLES[kind]
    one, batch = st.tabs(["One record", "Batch (CSV)"])
    with one:
        with st.form(f"form_{kind}"):
            rec = {n: _widget(kind, n, m, sample.get(n)) for i, (n, m) in enumerate(fields.items())}
            go = st.form_submit_button(f"Check {TITLES[kind].lower()} record", type="primary")
        st.caption("Pre-filled values are the Task 8 example record, which is deliberately unusual. Replace them with real values.")
        if go and alerts.single_flight("anom_busy"):
            rec = {k: v for k, v in rec.items() if v not in ("", None) or fields[k].get("required")}
            try:
                with st.spinner("Checking…"):
                    st.session_state[f"anom_one_{kind}"] = client.anomaly(kind, [rec])
            except ApiError as e:
                st.session_state.pop(f"anom_one_{kind}", None)
                alerts.show_error(e)
            finally:
                alerts.release("anom_busy")
        if f"anom_one_{kind}" in st.session_state:
            _show(f"one_{kind}", st.session_state[f"anom_one_{kind}"])
        elif not go:
            alerts.empty_state("No record checked yet.")
    with batch:
        tpl = pd.DataFrame([{k: v for k, v in sample.items()}])
        st.download_button("Download CSV template", tpl.to_csv(index=False).encode(), f"{kind}_template.csv", "text/csv", key=f"tpl_{kind}")
        st.caption(f"Columns must match the template. Up to {uploads.MAX_ROWS} rows. Dates as YYYY-MM-DD.")
        up = st.file_uploader("CSV file", type=["csv"], key=f"csv_{kind}")
        if up is not None and st.button("Check batch", type="primary", key=f"btn_{kind}") and alerts.single_flight("anom_busy"):
            try:
                recs = uploads.table_to_records(uploads.read_csv_upload(up.name, up.getvalue()), fields)
                with st.spinner("Checking…"):
                    st.session_state[f"anom_batch_{kind}"] = client.anomaly(kind, recs)
            except uploads.UploadError as e:
                st.session_state.pop(f"anom_batch_{kind}", None)
                st.error(str(e))
            except ApiError as e:
                st.session_state.pop(f"anom_batch_{kind}", None)
                alerts.show_error(e)
            finally:
                alerts.release("anom_busy")
        if f"anom_batch_{kind}" in st.session_state:
            _show(f"batch_{kind}", st.session_state[f"anom_batch_{kind}"])
        elif up is None:
            alerts.empty_state("Upload a CSV to check many records.")


def render(client, cfg):
    st.header("Anomaly Detection")
    st.info("An anomaly is an **investigation signal**, not proof of employee misconduct or payroll fraud.")
    t1, t2 = st.tabs(["Attendance", "Payroll"])
    with t1:
        _kind(client, "attendance")
    with t2:
        _kind(client, "payroll")
