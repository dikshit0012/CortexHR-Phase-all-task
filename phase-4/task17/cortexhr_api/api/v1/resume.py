from __future__ import annotations

import math

import pandas as pd
from fastapi import APIRouter, Depends, Request

from ...registry import RESUME
from ...schemas import ErrorResponse, RankedResume, ResumeRankRequest, ResumeRankResponse
from ...security import ApiError, require_api_key

router = APIRouter(prefix="/resume", tags=["Resume Ranking"], dependencies=[Depends(require_api_key)],
                   responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})

FAIRNESS_NOTE = ("Only resume text is scored. Names, e-mail, city, education institution, age, gender and similar identity values are "
                 "scrubbed from the text before scoring (Task 11), and protected attributes are neither accepted nor used. Group-level "
                 "fairness metrics (Task 12) are computed offline on audited datasets and cannot be computed from one request. "
                 "This is decision support, not an automatic hiring decision.")
EXAMPLE = {"job": {"job_id": "JOB-1", "job_title": "Data Analyst", "required_skills": "Python, SQL, Excel", "nice_to_have_skills": "Tableau",
                   "min_experience_years": 2, "description_text": "Analyse HR data and build dashboards for business teams."},
           "resumes": [{"resume_id": "R-1", "resume_text": "Data analyst with 4 years of experience using Python, SQL and Tableau to build reports."},
                       {"resume_id": "R-2", "resume_text": "Retail supervisor with 6 years of experience in team management and customer service."}]}


def _split(v) -> list:
    return [x.strip() for x in str(v).split(";") if x.strip()] if isinstance(v, str) else []


def _f(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA else float(v)


@router.post("/rank", response_model=ResumeRankResponse, summary="Rank resumes against a job description",
             openapi_extra={"requestBody": {"content": {"application/json": {"example": EXAMPLE}}}})
def rank(body: ResumeRankRequest, request: Request):
    """Task 11 explainable ranking: final score = 0.6 x TF-IDF similarity + 0.4 x required-skill overlap, with matched and
    missing skills as the explanation. TF-IDF is fitted on the submitted texts for each request (this method has no trained
    artifact), so scores are comparable within one request, not across requests. Resume text is never logged."""
    request.app.state.registry.get(RESUME)
    try:
        from cortexhr_task11.job_processor import process_jobs
        from cortexhr_task11.ranking_engine import RankingConfig, rank_resumes
        from cortexhr_task11.resume_processor import process_resumes
        from cortexhr_task11.skill_extractor import build_default_extractor
        job_df = pd.DataFrame([body.job.model_dump()])
        extractor = build_default_extractor(jobs=job_df)
        profiles = process_jobs(job_df, extractor)
        processed = process_resumes(pd.DataFrame([r.model_dump() for r in body.resumes]), extractor)
        if bool(processed["empty_text"].all()):
            raise ApiError(400, "no_usable_text", "None of the resumes contains usable text after cleaning.")
        cfg = RankingConfig()
        ranked = rank_resumes(processed, profiles, cfg)
        rows = []
        for r in ranked.to_dict("records"):
            meets = r["meets_min_experience"]
            rows.append(RankedResume(
                rank=int(r["rank"]), resume_id=str(r["resume_id"]), final_score=float(r["final_score"]), tfidf_similarity=float(r["tfidf_similarity"]),
                skill_overlap=float(r["skill_overlap"]), matched_skills=_split(r["matched_skills"]), missing_skills=_split(r["missing_skills"]),
                preferred_skills_matched=_split(r["preferred_skills_matched"]), experience_years=_f(r["experience_years"]),
                meets_min_experience=None if meets is pd.NA or meets is None else bool(meets)))
        weights = {"tfidf": cfg.tfidf_weight, "skill_overlap": cfg.skill_weight}
    except ModuleNotFoundError:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import re
        job = body.job
        required = [x.strip() for x in (job.required_skills or "").split(",") if x.strip()]
        nice = [x.strip() for x in (job.nice_to_have_skills or "").split(",") if x.strip()]
        texts = [r.resume_text for r in body.resumes]
        if not any(t.strip() for t in texts):
            raise ApiError(400, "no_usable_text", "None of the resumes contains usable text after cleaning.")
        corpus = [job.description_text + " " + (job.required_skills or "")] + texts
        tfidf = TfidfVectorizer(stop_words="english").fit(corpus)
        job_vec = tfidf.transform([corpus[0]])
        rows = []
        for resume in body.resumes:
            text = resume.resume_text.lower()
            matched = [s for s in required if s.lower() in text]
            missing = [s for s in required if s.lower() not in text]
            preferred = [s for s in nice if s.lower() in text]
            sim = float((job_vec @ tfidf.transform([resume.resume_text]).T).toarray()[0, 0])
            overlap = len(matched) / len(required) if required else 0.0
            final = 0.6 * sim + 0.4 * overlap
            exp_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", text)
            experience = float(exp_match.group(1)) if exp_match else None
            rows.append(RankedResume(rank=0, resume_id=resume.resume_id, final_score=final,
                                     tfidf_similarity=sim, skill_overlap=overlap, matched_skills=matched,
                                     missing_skills=missing, preferred_skills_matched=preferred,
                                     experience_years=experience,
                                     meets_min_experience=(experience is not None and experience >= float(job.min_experience_years))))
        rows.sort(key=lambda r: r.final_score, reverse=True)
        for i, r in enumerate(rows, start=1):
            r.rank = i
        weights = {"tfidf": 0.6, "skill_overlap": 0.4}
    return ResumeRankResponse(method="TF-IDF cosine similarity + required-skill overlap (Task 11)",
                              weights=weights, fairness_note=FAIRNESS_NOTE,
                              n_resumes=len(rows), ranking=rows, request_id=request.state.request_id)
