# Model Card: Resume fairness evaluation and fair screener (Task 12)

## 1. Name
Counterfactual audit of the Task 11 screener and the "Task 12 fair screener" (masks candidate name, e-mail/URL, location and institution tier before scoring). Not a separate predictive model.

## 2-4. Overview, use, out of scope
Detects text-based bias pathways in resume scoring. Out of scope: certifying a screener as "unbiased"; legal compliance sign-off.

## 5. Data
25 counterfactual twin pairs (`resume_bias_pairs`: 13 change name + college tier, 12 change name only); protected attributes file `unlock_CX-12` (inferred gender and college tier for 185 resumes) used only for aggregate auditing. **Genders are inferred from names, not self-reported.**

## 7/8. Results (re-run with the real protected-attribute file)
| Screener | Avg twin gap | Max | Non-zero pairs |
|---|---|---|---|
| Task 11 as shipped | 0.000000 | 0.000000 | 0 |
| Task 11 text only | 0.000180 | 0.000677 | 25 |
| Task 12 fair | 0.000000 | 0.000000 | 0 |
Stress test (synthetic, 31 resumes): when age/date of birth is disclosed, text-only Task 11 shows mean gap 0.0082, max 0.0146, **10 of 31 above the 0.01 tolerance**; the fair screener 0 on all perturbations.
Group level, fair screener, 185 resumes: gender group means Female 0.379 vs Male 0.365 (gap 0.0138); college tier Tier 1 0.385, Tier 2 0.350, Tier 3 0.383 (gap 0.0344). Job-stratified gender gap averages 0.041 across 10 jobs, largest 0.069 (J02, 3 vs 5 candidates).

## 9. Limitations
Only 25 twins, mostly gender-coded names; no race, caste, disability or intersectional tests; synthetic perturbations are not evidence on real applicants; zero twin gap does **not** prove fairness; job-level groups are tiny; name masking is a header heuristic.

## 10. Ethical considerations
Equal scores for twins say nothing about whether skill keywords themselves encode bias. Decide accuracy-fairness trade-offs through governance, not by the builder alone.

## 11/12. Deployment and monitoring
Re-run weekly by `monitor.py` (twin check OK, 25/25). Not deployed as a separate service.

## 13. Release recommendation
**RELEASE WITH WARNINGS** (as an audit tool; masking recommended for any real screening use).
