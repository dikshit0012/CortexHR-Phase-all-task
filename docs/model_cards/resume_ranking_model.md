# Model Card: Resume ranking (`task11-tfidf-skill-overlap-1.0`)

## 1. Name / version
Task 11 hybrid ranker; API method string `TF-IDF cosine similarity + required-skill overlap`, weights 0.6 / 0.4. **This is a fixed formula, not a trained model** (weights are not tuned).

## 2. Overview
Scores resumes against a job description to produce a shortlist ordering for a recruiter.

## 3. Intended use
Support a human recruiter who reads the shortlist and the unranked remainder.

## 4. Out-of-scope use
Automatic rejection or hiring; ranking without human review; use on languages/formats outside English text resumes; use as evidence about candidate quality or potential.

## 5. Dataset
`job_descriptions` (10 jobs, J01-J10) and `resumes` (185 resumes in the audited set), 555 human-labelled resume-job pairs (`resume_job_labels`; labels used for evaluation only). Fictional data.

## 6. Methodology
Text preprocessing, TF-IDF cosine similarity (60%) + share of required skills present (40%). No training.

## 7. Evaluation (re-run, 10 jobs, k = 5)
| Method | Top-5 hits | Mean P@5 | Mean R@5 | Mean NDCG@5 |
|---|---|---|---|---|
| TF-IDF only | 45 | 0.90 | 0.545 | 0.854 |
| Skill overlap only | 45 | 0.90 | 0.527 | 0.870 |
| **Hybrid 60/40** | 46 | **0.92** | 0.538 | **0.912** |
Random expectation for precision@5: 0.168. 16 misrankings (8 false shortlist, 8 missed shortlist). Exact score ties in 500 rows (10 inside a labelled top-k). Spearman score vs experience 0.078 (p = 0.067).

## 8. Fairness evaluation
See `resume_fairness_evaluation.md`. Counterfactual twins: 25 pairs, shipped gap 0.000, tolerance 0.01.

## 9. Limitations
Small evaluation (10 jobs, ~55 labelled resumes each); weights untuned; vocabulary overlap favours keyword-heavy resumes; ties; human labels may be biased.

## 10. Ethical considerations
Keyword matching can disadvantage non-standard resumes and reproduce historic hiring patterns. A human must review every candidate regardless of rank.

## 11. Deployment
`POST /v1/resume/rank`; verified locally (200 for 1 job x 6 resumes). Not tested remotely.

## 12. Monitoring
Twin-resume check re-run by Task 18: 25/25 pairs, avg and max gap 0.000 (status OK).

## 13. Release recommendation
**RELEASE WITH WARNINGS.**
