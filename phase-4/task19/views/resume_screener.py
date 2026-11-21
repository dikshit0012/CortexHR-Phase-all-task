import re

import pandas as pd
import streamlit as st

from api_client import ApiError
from components import alerts, charts, uploads


def _collect(pasted: str, files) -> tuple[list[tuple[str, str]], list[str]]:
    """Return [(local label, text)] and a list of problems. Filenames stay in the browser: only generic ids are sent."""
    items, problems = [], []
    for i, chunk in enumerate([c.strip() for c in re.split(r"(?m)^\s*---+\s*$", pasted or "") if c.strip()], start=1):
        if len(chunk) > uploads.MAX_RESUME_CHARS:
            problems.append(f"Pasted resume {i} is longer than {uploads.MAX_RESUME_CHARS:,} characters.")
        else:
            items.append((f"Pasted resume {i}", chunk))
    for f in files or []:
        try:
            items.append((f.name, uploads.read_resume_txt(f.name, f.getvalue())))
        except uploads.UploadError as e:
            problems.append(str(e))
    if len(items) > uploads.MAX_RESUMES:
        problems.append(f"At most {uploads.MAX_RESUMES} resumes per request.")
    return items, problems


def _show(res: dict, labels: dict[str, str]):
    df = pd.DataFrame(res["ranking"])
    df["label"] = df["resume_id"].map(labels).fillna(df["resume_id"])
    view = pd.DataFrame({
        "Rank": df["rank"], "Candidate": df["label"], "Score": df["final_score"].round(3),
        "Text similarity": df["tfidf_similarity"].round(3), "Required-skill overlap": df["skill_overlap"].round(3),
        "Matched skills": df["matched_skills"].map(", ".join), "Missing skills": df["missing_skills"].map(", ".join),
        "Preferred skills matched": df["preferred_skills_matched"].map(", ".join),
        "Experience (years)": df["experience_years"],
        "Meets minimum experience": df["meets_min_experience"].map({True: "Yes", False: "No"}).fillna("n/a")})
    st.dataframe(view, width="stretch", hide_index=True)
    st.plotly_chart(charts.resume_scores(df[["label", "final_score"]]), width="stretch")
    w = res.get("weights") or {}
    st.caption(f"{res.get('method', '')}. Weights: text similarity {w.get('tfidf', 0):.0%}, required skills {w.get('skill_overlap', 0):.0%}. "
               "Scores are comparable only within this request, not across requests.")
    with st.expander("How fairness is handled"):
        st.write(res.get("fairness_note", ""))
    alerts.review_label("resume")
    st.warning("AI-generated ranking – human review required. Do not reject candidates automatically.")


def render(client, cfg):
    st.header("Resume Screener")
    st.write("Rank candidates against a job description so a recruiter knows whom to read first. Names, e-mail and similar identity details are scrubbed before scoring.")
    st.info("Resume ranking is decision support, not a hiring decision. Candidates must never be rejected automatically.")
    with st.form("resume_form"):
        st.subheader("Job")
        title = st.text_input("Job title *", max_chars=200)
        c1, c2 = st.columns(2)
        req = c1.text_input("Required skills (comma separated)", max_chars=1000, placeholder="Python, SQL, Excel")
        nice = c2.text_input("Nice-to-have skills", max_chars=1000, placeholder="Tableau")
        minexp = st.number_input("Minimum years of experience (0 = none)", 0.0, 60.0, 0.0, step=0.5)
        desc = st.text_area("Job description (optional)", max_chars=10_000, height=120)
        st.subheader("Candidates")
        pasted = st.text_area("Paste resumes here, separated by a line containing only ---", height=180)
        files = st.file_uploader("…or upload .txt resumes", type=["txt"], accept_multiple_files=True)
        st.caption("Only plain-text (.txt, UTF-8) resumes are supported by the API; PDF and Word files are not. Up to 50 resumes, 20,000 characters each.")
        go = st.form_submit_button("Rank candidates", type="primary")
    if go:
        items, problems = _collect(pasted, files)
        for p in problems:
            st.error(p)
        if not title.strip():
            st.warning("Please provide all required information: enter a job title.")
        elif not items:
            st.warning("Please add at least one resume.")
        elif not problems and alerts.single_flight("resume_busy"):
            job = {"job_title": title.strip(), "description_text": desc, "required_skills": req, "nice_to_have_skills": nice}
            if minexp > 0:
                job["min_experience_years"] = float(minexp)
            ids = {f"C-{i}": label for i, (label, _) in enumerate(items, start=1)}
            try:
                with st.spinner("Ranking…"):
                    res = client.resume_rank(job, [{"resume_id": cid, "resume_text": t} for cid, (_, t) in zip(ids, items)])
                st.session_state["resume_res"] = (res, ids)
            except ApiError as e:
                st.session_state.pop("resume_res", None)
                alerts.show_error(e)
            finally:
                alerts.release("resume_busy")
    if "resume_res" in st.session_state:
        _show(*st.session_state["resume_res"])
    elif not go:
        alerts.empty_state('No ranking yet. Enter a job and at least one resume, then press "Rank candidates".')
