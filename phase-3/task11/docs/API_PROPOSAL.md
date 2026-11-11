# Proposed resume-ranking endpoint (not implemented)

This is an API design only. Task 11 does **not** add a router or a running HTTP endpoint to the CortexHR service. The current
`ranking_engine.top_k_for_job(...)` helper formats results from an already-computed ranking; it does not accept a request, load a
model, or rank arbitrary candidate input by itself.

## Endpoint

`POST /v1/resume/rank`

The request supplies either an existing `job_id` or an inline `job_description`, plus the candidate resume texts to rank.
Exactly one job form is required.

### Request using a stored job

```json
{
  "job_id": "J04",
  "candidates": [
    {"resume_id": "candidate-001", "resume_text": "Python and machine learning experience..."},
    {"resume_id": "candidate-002", "resume_text": "SQL and model deployment experience..."}
  ],
  "top_k": 5
}
```

### Request using an inline job description

```json
{
  "job_description": {
    "job_title": "ML Engineer",
    "required_skills": ["Python", "Machine Learning", "SQL"],
    "nice_to_have_skills": ["Model Deployment"],
    "description_text": "Build and deploy machine-learning systems."
  },
  "candidates": [
    {"resume_id": "candidate-001", "resume_text": "Python and machine learning experience..."}
  ],
  "top_k": 5
}
```

| Field | Type | Validation |
|---|---|---|
| `job_id` | string | Existing job identifier; mutually exclusive with `job_description` |
| `job_description` | object | Inline title, required/preferred skills, and description; mutually exclusive with `job_id` |
| `candidates` | array | Required; 1..1000 objects with a unique, non-identifying `resume_id` and non-empty `resume_text` |
| `top_k` | integer | Optional; default 5; range 1..100 |

Names, e-mail addresses, city, age/date of birth, gender/pronouns, college tier, and institution names are not accepted as
structured candidate fields. The resume text is passed through the same redaction step used by the ranking pipeline. Unknown
request fields should be rejected; human labels must never be accepted. Text redaction is pattern-based and is not a universal
PII detector; production use needs a separately validated privacy layer.

## Response 200

The response contains the ordered top results. This is a proposed shape, not a captured live response:

```json
{
  "job_id": "J04",
  "job_title": "ML Engineer",
  "model_version": "task11-tfidf-v1",
  "ranked_candidates": [
    {
      "resume_id": "candidate-001",
      "rank": 1,
      "final_score": 0.542,
      "tfidf_similarity": 0.237,
      "skill_overlap": 1.0,
      "matched_skills": ["Python", "Machine Learning", "SQL"],
      "missing_skills": [],
      "explanation": {
        "tfidf_top_terms": ["machine learning", "python"],
        "tfidf_weighted_contribution": 0.1422,
        "skill_overlap_weighted_contribution": 0.4
      },
      "experience_years": 6.7
    }
  ],
  "decision_support_notice": "Advisory ranking only. A human must review every candidate."
}
```

`final_score = 0.60 * tfidf_similarity + 0.40 * skill_overlap`. `skill_overlap` is the matched required-skill count divided by
the number of required skills. Preferred skills are not counted in that overlap, but are included in the job text used for TF-IDF
and can therefore affect `tfidf_similarity`. The explanation reports matched
and missing required skills and the weighted score components; TF-IDF terms are informational, not causal proof. Experience may
be shown when stated, but must not affect the score. Scores are in `[0, 1]`; ties are broken deterministically by `resume_id`.

## Errors

| Status | When |
|---|---|
| 404 | Unknown `job_id` |
| 413 | Request exceeds the configured candidate or text-size limit |
| 422 | Missing/both job inputs, invalid `top_k`, empty candidate list/text, duplicate IDs, or disallowed fields |
| 503 | Ranking model or required job data is unavailable |

## Implementation requirements

The current offline pipeline fits TF-IDF on its resume/job corpus. A service must load a versioned, fixed vectorizer and skill
dictionary (or explicitly version and rebuild them); refitting per request would make candidate scores change with the request
contents. It must scrub before scoring, log only request metadata (not resume text), retain an access audit trail, and show the
advisory notice. Re-run the fairness audit after changing the candidate pool, vectorizer, or skill dictionary.
