# Task 12 - Audit your screener for fairness (CortexHR, Phase 3)

Counterfactual (twin-resume) audit of the Task 11 resume screener, a text-only **fair screener** that masks personal information, and a before/after
comparison of fairness **and** ranking quality. The ranking is a decision-support tool, not an automatic hiring decision. The full fairness statement is in
[`reports/fairness_summary.md`](reports/fairness_summary.md) (generated from the results, not typed by hand).

## 1. Objective
Measure whether the Task 11 screener scores otherwise identical resumes differently when only a protected attribute changes, reduce any such sensitivity without
removing job-relevant information, and verify ranking quality is preserved.

## Run locally
Use Python 3.10 or newer and install the Task 11/12 dependencies in a dedicated environment:

```powershell
python -m pip install -r phase-3\task11\requirements.txt
python -m pip install -r phase-3\task12\requirements.txt
```

The Task 11 source is included, but its `job_descriptions.csv` and `resumes.csv`, plus Task 12's `resume_bias_pairs.csv`, must be obtained through an approved channel and stored under `phase-3\task11\data\` and `phase-3\task12\data\` (or configured with `CORTEXHR_TASK11_DATA` and `CORTEXHR_TASK12_PAIRS`). The protected-attribute input is optional and confidential; if supplied, keep it under the ignored `phase-3\task12\private\` directory or set `CORTEXHR_TASK12_PRIVATE`. Do not add candidate data or protected attributes to Git.

Run unit/integration tests and the audit after installing the data:

```powershell
python -m pytest phase-3\task12\tests
python phase-3\task12\scripts\run_task12.py
```

The data-backed test module reports a clear skip when its Task 11 and twin-pair CSV inputs are absent. The protected-attribute comparison is omitted when the confidential file is not provided.

## 2. Data (inspected)
| item | content |
|---|---|
| `data/resume_bias_pairs.csv` (workbook sheet `resume_bias_pairs`) | 25 pairs: `pair_id`, `resume_id_original`, `resume_id_twin`, `attribute_changed`, `skills_identical`=Y, `experience_identical`=Y, `expected_score_difference`=0. 13 pairs change *name + college tier*, 12 change *name only*. |
| `unlock_CX-12_resume_protected_attrs.csv` (confidential, **git-ignored**, `private/`) | `resume_id`, `inferred_gender` (Male 111 / Female 74), `college_tier` (Tier 1/2/3) for all 185 resumes. **No age / date of birth.** |
| `cortexhr_helpers.py` | course helper copy in `src/cortexhr_phase-3/task12/`; `group_gap()` returns unweighted group means and their range, excluding incomplete rows and rejecting unusable input. |
| Task 11 data | the 25 twin resumes are rows RES-0161...RES-0185 of `resumes.csv`; originals are earlier ids. Read from `../task11/data`. |

The workbook README mentions `protected_attributes` / `resume_protected_attrs` sheets; they are absent from the supplied workbook, so the supplied unlock file is the only protected-attribute source.
Twin construction (verified by line diff): the twins differ **only** in the header name, the name-bearing e-mail and, for 13 pairs, the "Tier N institution" line - 0 other differing lines.
Gender direction: 14 pairs Female->Male, 11 Male->Female. Tier changed in 13 pairs (11 twin higher tier, 2 lower).

## 3. Method
* **Twin gap** = |score_A - score_B| on the Task 11 final score (0-1). The CSV records both signed `score_B - score_A` and absolute gaps. Primary run: the job each pair targets (`target_job_family`, audit set-up only). Robustness run: all 10 jobs (250 comparisons).
* **Tolerance** (documented): 0.01 = one point on a 0-100 view. Non-zero uses epsilon 1e-9.
* Three screeners, all using the *unchanged* Task 11 engine (`rank_resumes`, skill extractor, TF-IDF + overlap 60/40):
  1. `task11_as_shipped` - the original Task 11 pipeline (scrubs name/e-mail/city/education/tier by looking up structured-column values).
  2. `task11_text_only` - the same Task 11 code when only `resume_id` + `resume_text` are available.
  3. `task12_fair` - text-only personal-information masking (`fair_preprocessing.py`), given only `resume_id` + `resume_text`.
* Protected values are used only to analyse results. Human labels are used only to evaluate ranking quality.

## 4. Baseline results (25 pairs, primary job)
| Screener | Avg gap | Median | Max | Min | Non-zero pairs | > tolerance |
|---|---:|---:|---:|---:|---:|---:|
| Task 11 as shipped | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 |
| Task 11 text-only | 0.000180 | 0.000152 | 0.000677 | 0.000004 | 25 | 0 |
| Task 12 fair | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 |

**Honest reading:** Task 11 as shipped was *already* neutral on these twins (gap exactly 0), so the gap reduction against it is **undefined (baseline 0)**.
When the structured columns are withheld, every pair shows a small non-zero gap (avg 0.000180; max 0.000677, ~15x below tolerance); the fair screener removes it (**100% reduction vs the text-only baseline**, but from a negligible level).
All 250 pair-job comparisons: text-only avg 0.000035, fair 0.
Direction (text-only): male-coded names scored higher in 13 pairs, lower in 12 (mean male - female +0.000054, sign test p = 1.0) - no systematic favouring, only noise-level sensitivity.

Comparison table (`reports/fairness_comparison.csv`, text-only -> fair):

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| Average twin gap | 0.000180 | 0.000000 | 100.0% |
| Median twin gap | 0.000152 | 0.000000 | 100.0% |
| Maximum twin gap | 0.000677 | 0.000000 | 100.0% |
| Minimum twin gap | 0.000004 | 0.000000 | 100.0% |
| Non-zero gap pairs | 25 | 0 | 25 fewer |

## 5. Bias sources (measured, `reports/bias_source_ablation.csv`)
| masking applied to the twins' text | mean gap | max gap | non-zero pairs |
|---|---:|---:|---:|
| nothing (raw) | 0.000463 | 0.002732 | 25 |
| e-mail only | 0.000179 | 0.000663 | 25 |
| institution tier only | 0.000467 | 0.002783 | 25 |
| name only | 0.000142 | 0.000387 | 21 |
| name + e-mail | 0.000000 | 0.000000 | 0 |
| all categories (fair) | 0.000000 | 0.000000 | 0 |

* **Names and name-bearing e-mails** cause the whole gap; either one alone leaves a residual, both together remove it. Mechanism: rare name tokens change the TF-IDF vector norm. All gap is in the TF-IDF component; skill-overlap gap is 0.
* **Skill extraction was consistent**: identical skills and experience for twins in all 250 comparisons (all screeners).
* **Tier did not leak, by accident:** Task 11 drops numeric tokens, so "Tier 1" and "Tier 2" are textually identical. A different wording would leak; the fair screener masks it explicitly.
* **No other text differences** between twins (0 lines). **Age, DOB, graduation years, pronouns, titles, phone, personal fields do not occur in the 185 resumes** (`text_risk_scan.csv`), so the twins cannot test them.
* Proxy check: gender and tier are not significantly associated with target job family (chi-square p = 0.82 and 0.64, small counts) - so legitimate skills were **not** removed to "fix" group differences.

## 6. Fairness improvements (`src/cortexhr_phase-3/task12/fair_preprocessing.py`)
General, text-only rules (no pair ids, no name lists): header candidate name, e-mail/URL, phone, `City |` contact prefix, institution-tier statements, age / date of birth, gender / marital / religion fields (and the "Personal details" heading),
gendered honorifics (`Ms.` with the dot - **"MS Excel" is kept**), pronouns, graduation years / "class of". **Kept:** skills, certifications, degree subjects, job titles, employers, the experience statement.
The fair screener sees only `resume_id` and `resume_text` (feature isolation); tests permute every structured/protected column and show scores do not move. The originals stay available for reproducible before/after runs. During this audit, `load_pairs()` was hardened to reject missing fields and enforce the expected 25-pair input; `score_pairs()` now records signed gaps and fails explicitly if either twin is absent. `group_gap()` now validates lengths and numeric values and handles missing observations pairwise instead of silently dropping them in pandas groupby.

Two issues found and fixed *during* the work, and why they matter:
* A first version deleted the whole tier line, which also removed the degree subject "Statistics" from 12 resumes (a legitimate skill). The rule now masks only the tier statement.
* The first synthetic stress test used symmetric token swaps and showed gaps of 0 even for the raw screener (equally rare tokens have identical norms); it was redesigned (disclosure vs non-disclosure, one resume at a time). The redesigned test then exposed a leftover "years old"/"PERSONAL" residue (gap 0.0043), which was fixed in the general rules.

**Synthetic stress test** (`reports/synthetic_stress_test.csv`; 31 real resumes, one at a time; *not* audit data on real candidates):

| perturbation | text-only mean / max gap | fair mean / max gap |
|---|---:|---:|
| age + date of birth disclosed | 0.0095 / 0.0168 (14 of 31 above tolerance) | 0 / 0 |
| gender + marital status disclosed | 0.0075 / 0.0133 (8 above) | 0 / 0 |
| graduation year old vs recent | 0.0015 / 0.0027 | 0 / 0 |
| title Mr. vs Ms. | 0.0010 / 0.0024 | 0 / 0 |
| pronouns he vs she | 0 / 0 (pronouns are stop-words) | 0 / 0 |

This shows the text-only Task 11 pipeline *can* exceed the tolerance when candidates disclose age or personal details - a risk the supplied twins could not reveal.

## 7. Ranking quality (same Task 11 labels and method; labels used only for evaluation)
| Metric | Task 11 as shipped | Task 12 fair | Difference |
|---|---:|---:|---:|
| Precision@5 | 0.9200 | 0.9200 | 0.0000 |
| Recall@5 | 0.5380 | 0.5380 | 0.0000 |
| NDCG@5 | 0.9115 | 0.9213 | +0.0097 |
| Top-5 Hits | 46 / 50 | 46 / 50 | 0 |

The fair screener keeps degree text that Task 11 scrubbed, so a small change is expected (top-5 sets differ in 3 of 10 jobs; the earlier whole-line version was identical to shipped). With 10 jobs and 95 relevant candidates the +0.0097 is within noise: **no meaningful ranking quality reduction; not evidence of an accuracy gain.**
Text-only Task 11: NDCG@5 0.9226 (fair - text-only = -0.0013).

## 8. Fairness statement (summary - full text in `reports/fairness_summary.md`)
Tested: 25 gender-coded twin pairs (13 also change college tier) on the Task 11 screener. Average twin gap: shipped 0.000000; text-only 0.000180 -> fair 0.000000 (100% vs text-only; undefined vs shipped).
Ranking quality did not meaningfully change. Limitations remain (below). **Who decides the trade-off?** Fairness and accuracy should not be treated as competing goals where accuracy automatically wins; the acceptable trade-off should be decided through
transparent governance with affected stakeholders, HR professionals, legal/compliance and responsible-AI experts. A small ranking loss may be acceptable if it meaningfully reduces discriminatory outcomes, provided it is evidence-based, documented and monitored.
**Equal accuracy does not mean equal treatment:** accuracy is a population average, treatment is how comparable individuals are scored.

## 9. Limitations
* 25 twins, one attribute pair per resume (gender-coded name, sometimes tier); gender is inferred; no age, race, caste, disability or intersectional twins; no age in the audit data.
* The supplied twins cannot test most masked pathways; the stress test is synthetic. Real resumes carry more proxies (school names, employers, hobbies, gaps, locations).
* Group-level gaps persist for the fair screener (all resumes: gender 0.0138, tier 0.0344 with Tier 2 lowest); they reflect different content and are not shown to be treatment effects, but must be monitored (`group_gap_report.csv`). Twin-only tier gap (0.137) is a composition artefact of 50 different resumes.
* A follow-up job-stratified audit (`job_stratified_group_gap.csv`) compares candidates only within their declared target job. With at least 3 resumes per group, the equal-job-weighted mean maximum score gap is 0.0413 for gender (10 jobs) and 0.1099 for college tier (9 jobs); the largest gaps are 0.0689 for gender in J02 and 0.2055 for tier in J08. These are descriptive small-sample comparisons, not evidence that the score caused the gap. Job-centered score differences move in the same direction as skill-overlap and human-suitability differences. Human labels can be biased too.
* No post-hoc group-specific score adjustment was made. This model ranks candidates using job-text similarity and required-skill overlap; applying a protected-group correction without evidence of model error could change treatment based on protected status. The measured intervention remains feature isolation and text masking. Review job-specific outcomes, extraction quality, and label consistency with HR/legal governance before considering any score-policy change.
* Name masking is a header heuristic; multi-line names or names in free text can be missed. Human labels may be biased. Tolerance 0.01 is a documented choice, not a legal standard.
* Task 11 correction discovered here: its 500 tied scores are exactly the 25 twin pairs x 10 jobs x 2 (README fixed).

## 10. Run
```
cd task12
python -m venv .venv && source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# place the confidential unlock_CX-12_resume_protected_attrs.csv in phase-3/task12/private/  (git-ignored; or set CORTEXHR_TASK12_PRIVATE)
python scripts/run_task12.py            # full synthetic stress test (31 resumes); add --stress-step 60 for a quicker run
python -m pytest -q                     # 50 tests
```
Requires `../task11` (code, data, reports). Overrides: `CORTEXHR_TASK11_ROOT`, `CORTEXHR_TASK11_DATA`, `CORTEXHR_TASK12_PAIRS`, `CORTEXHR_TASK12_PRIVATE`, `CORTEXHR_TASK12_REPORTS`.
Without the confidential file the score audit still runs; protected-value analyses are skipped and reported as unavailable.

## 11. Outputs - `reports/`
`twin_gap_before.csv` and `twin_gap_after.csv` (one row per pair, scores, signed and absolute gaps), `fairness_comparison.csv`, `ranking_quality_comparison.csv`, `fairness_audit_report.json`, `fairness_statement.md`, and `twin_gap_comparison.svg`.
The original expanded reports are retained: `fairness_before.csv` (both before-screeners, 25 pairs each), `fairness_after.csv`, `fairness_summary.md`, `fairness_all_jobs_250_comparisons.csv`, `bias_source_ablation.csv`, `twin_text_differences.csv`, `text_risk_scan.csv`, `synthetic_stress_test.csv`, `protected_attribute_analysis.csv` (aggregated), `group_gap_report.csv`, `job_stratified_group_gap.csv` (aggregated by target job and protected group; no resume IDs), `proxy_scan.csv`, and `run_summary.json`.
`reports/private/protected_pair_details.csv` holds per-pair gender/tier values and is git-ignored.

## 12. Technologies
Python 3.12, pandas, NumPy, scikit-learn (via Task 11), SciPy (chi-square, sign test), pytest.
