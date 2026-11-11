# Task 11 - Resume ranking

Task 11 ranks resumes against job descriptions using extracted skills and experience, then evaluates the ranking against human labels. The source and synthetic tests are included, but the three CSV inputs are not.

## Setup and inputs

Use Python 3.10 or newer. Install dependencies and run the synthetic/unit tests from the repository root:

```powershell
python -m pip install -r phase-3\task11\requirements.txt
python -m pytest phase-3\task11\tests
```

For an end-to-end run, obtain the three approved CSVs and place them in `phase-3\task11\data\`:

| File | Required columns |
|---|---|
| `job_descriptions.csv` | `job_id`, `job_title`, `min_experience_years`, `required_skills`, `nice_to_have_skills`, `description_text` |
| `resumes.csv` | `resume_id`, `resume_text` |
| `resume_job_labels.csv` | `resume_id`, `job_id`, `human_suitability_score` (1-5), `shortlist` (`Y`/`N`) |

These are candidate-level records. Keep them and generated reports in approved, access-controlled storage; `phase-3\task11\data\` and `phase-3\task11\reports\` are Git-ignored.

Run the pipeline after the files are in place:

```powershell
python phase-3\task11\scripts\run_task11.py
```

Alternatively, pass `--data-dir` and `--reports-dir` to use approved locations outside the checkout. The `CORTEXHR_TASK11_DATA` and `CORTEXHR_TASK11_REPORTS` environment variables are also supported. The required columns, validation, and defaults are defined in `src\cortexhr_phase-3\task11\data_loading.py` and `paths.py`.
