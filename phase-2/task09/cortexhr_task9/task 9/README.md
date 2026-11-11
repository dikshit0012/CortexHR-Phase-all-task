# CortexHR Task 9 - HR Intent Classification, FAQ Answers and a Safe Fallback

An HR chatbot brain for Zenith Nova Technologies (part of Aura Workforce): it reads an employee's question, decides which of 33 intents it is, answers from the **approved HR FAQ** when it is confident and an approved answer exists, and otherwise says it is not sure. It is added to the Task 8 FastAPI service as `POST /v1/chat`.

> This is an **intent-classification service built on a traditional machine-learning model** (TF-IDF + Logistic Regression). It is **not a large language model**: it cannot write new answers, it can only pick one of 33 trained intents and return text that already exists in `hr_faq.csv`.

**Status: works, with documented limits. Not production-ready** (see [Known limitations](#known-limitations)). Every number below comes from a real run whose raw output is in `reports/`.

---

## 1. Objective

Train an intent classifier on `chatbot_nlu_training`, evaluate it on `chatbot_test_set`, answer FAQ intents from `hr_faq`, refuse safely when unsure, and expose it as `/v1/chat` in the Task 8 service. A wrong confident answer is worse than "I am not sure", so most design decisions favour refusing.

## 2. How it works (architecture)

```
POST /v1/chat {"message": "..."}                      (Task 8 app, task 8/app/chat.py)
        |  Pydantic validation (422 on bad input; empty / whitespace text is valid -> fallback)
        v
  ChatService.reply                                    (task 9/src/cortexhr_task9/chat_service.py)
   1. empty message?                          -> fallback
   2. Pipeline [clean -> TF-IDF -> LogReg]    -> intent + top probability   (model loaded ONCE at startup)
   3. error while predicting?                 -> fallback (details only in the log, type only)
   4. probability < threshold (0.64)?         -> fallback, intent = null
   5. intent == out_of_scope?                 -> fallback
   6. FAQ rules for this intent match?        -> verbatim answer from hr_faq.csv
   7. otherwise                               -> "no approved answer" fallback (intent is returned)
```

Classification (step 2) and answer lookup (step 6, `faq.py`) are separate modules: the FAQ code never sees a probability and the classifier never sees an answer.

### TF-IDF and Logistic Regression in plain English

TF-IDF converts text into numerical features based on how important words and phrases are in a message. Common words usually receive less weight, while words that help distinguish one message from another can receive more weight. Logistic Regression then uses those numbers to estimate how well the message matches each of the 33 intents and picks the best one. The "confidence" is the winning intent's share of that estimate. It is **not** a calibrated probability (see section 8).

## 3. Files

```
task 9/
  README.md
  pytest.ini
  data/                      chatbot_nlu_training.csv, chatbot_test_set.csv, hr_faq.csv   (copies of the starter files)
  src/cortexhr_task9/
    text_cleaning.py         clean_text() - same function in training and inference
    typo_corrections.json    110 reviewed typo fixes (built by scripts/build_typo_map.py)
    data.py                  loading + column validation, phrase groups for cross-validation
    intent_model.py          build / save / LOAD+VERIFY the versioned artifact, predict
    training.py              grouped cross-validation, config comparison, threshold selection
    faq.py                   FAQ loading, intent->FAQ rule engine, load-time validation
    intent_faq_map.json      the hand-reviewed intent -> FAQ rules (see section 7)
    chat_service.py          orchestration + fallback logic
    paths.py                 default paths (relative) and environment overrides
  scripts/                   train_intent_model.py, evaluate_intent_model.py, check_faq_coverage.py,
                             check_faq_questions.py, make_example_conversations.py, build_typo_map.py
  models/                    hr_intent_classifier.joblib (+ .metadata.json)   <- git-ignored, rebuild locally
  reports/                   all evidence (section 14)
  tests/                     152 pytest tests
task 8/app/chat.py           NEW  the /v1/chat router + schemas (HTTP layer only)
task 8/app/main.py           CHANGED (+9/-2 lines): include the chat router, load the chat model once in lifespan
```

Tasks 5-7 were not touched. Task 8 changed only in `main.py` (additive; see `git diff`) plus the new `chat.py`.

## 4. The data (inspected before training)

| | training | test | FAQ |
|---|---|---|---|
| file | `chatbot_nlu_training.csv` | `chatbot_test_set.csv` | `hr_faq.csv` |
| rows | 858 | 132 | 30 |
| columns | `example_id, text, intent, entities` | `test_id, text, expected_intent, expected_entities` | `faq_id, question, answer, category, policy_source` |
| intents | 33, exactly **26 messages each** | the same 33, **4 each** | none (no intent column) |

Findings (all measured, see `reports/dev_per_intent_accuracy.csv`):

* **No missing or empty messages**, no whitespace problems. 323 messages contain capital letters.
* **Duplicates:** 7 distinct texts appear twice (14 rows). **No conflicting labels** anywhere.
* **Train/test overlap: none.** 0 of 132 test messages equal a training message after cleaning, and 0 share a base phrase after removing filler words. The official test set is therefore *not contaminated*, so its score is a genuine generalisation measure.
* **The training data is very repetitive.** 26 messages per intent, but after removing filler (`pls`, `urgently`, `quick question`, `I want to know`, `hey`...) and fixing typos there are only **270 distinct base phrasings for 33 intents (median 6 per intent; the smallest has 4)**. The test set uses *new wording* ("cheers", "yo", "who works under me", "how do I quit the company", "will it rain tomorrow"). This is the single most important fact about the results.
* **Messy text I read** (45 random messages plus the distinct phrasings of every intent): filler prefixes/suffixes, informal words ("pls", "hey"), and mostly **adjacent-letter swap typos** ("laeve", "urgenlty", "thnaks", "attnedance", "paylsip", "slaary"). Month names, employee names ("Priya Nair", "Karthik") and numbers also appear.
* **Class balance:** perfectly balanced (26 each), so accuracy and macro-F1 are not distorted by imbalance. The test set is balanced too, but with only 4 per intent each intent's score is very noisy.
* **`hr_faq` has no intent column** - the link from an intent to an FAQ answer does not exist in the data, so I had to create and validate it (section 7). The only contact detail in the whole FAQ is `payroll@znova-demo.example` (FAQ-28, *payroll queries only*).

## 5. Text cleaning (`text_cleaning.py`)

* Unicode NFKC, lowercase, trim, collapse repeated whitespace, curly quotes to straight, collapse `???` to `?`.
* **Targeted typo corrections only:** 110 pairs derived from the *training* vocabulary using one rule (a rare token that is one adjacent-letter swap away from a more frequent token), then **reviewed by hand**. One false positive was removed (`form -> from` would have destroyed "Form 16") and one typo the rule cannot find was added (`makrs -> marks`). Unknown words are never "corrected".
* **Nothing meaningful is removed:** no stop-word list, so negations, numbers, dates, names and `form 16` survive. Filler words are kept too: the model learns that they carry no signal, and removing them could erase real signal.
* The function never raises (None / NaN / numbers / bytes are handled) and is the **first step of the saved pipeline**, so training and inference cannot drift apart. 19 unit tests.

## 6. Model

`Pipeline([clean_texts -> FeatureUnion(TF-IDF word 1-2 grams, TF-IDF character 2-5 grams) -> LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42)])`, saved as **one** joblib artifact `models/hr_intent_classifier.joblib` (752 KB) with `models/hr_intent_classifier.metadata.json` next to it. Trained on `chatbot_nlu_training.csv` only (858 rows). The test set is not read by any training code (a test checks this, and that the artifact's recorded training-file hash differs from the test file's).

### Where I deviated from the brief's baseline, and why

The brief's starting point was word n-grams only with `C=1`. I measured it on a **development split of the training data**: 5-fold cross-validation grouped by base phrasing, so every dev message is a phrasing the model has not seen - the same situation as the test set (`reports/model_config_comparison_dev.csv`):

| configuration | dev accuracy | dev macro-F1 | accuracy with 1 injected typo | mean top probability | share answered at 0.60 |
|---|---|---|---|---|---|
| word only, C=1 (brief baseline) | 0.548 | 0.545 | 0.477 | 0.151 | **0.5%** |
| word only, C=10 | 0.550 | 0.546 | 0.479 | 0.393 | 25.9% |
| word + char, C=1 | 0.607 | 0.601 | 0.556 | 0.258 | 5.0% |
| **word + char, C=10 (chosen)** | 0.598 | 0.593 | 0.548 | 0.487 | 39.7% |
| word + char, C=30 | 0.599 | 0.592 | 0.549 | 0.562 | 47.7% |

* **Character n-grams** add about 5 accuracy points on unseen phrasings and about 7 on messages with typos the dictionary does not know (the typo test injects a random swap *after* cleaning, so the dictionary cannot hide it).
* **`C=1` makes the probabilities almost useless for a threshold:** with the brief's `C=1`, only 0.5% of dev messages would ever reach 0.60, so the bot would refuse nearly everything. `C=10` gives probabilities a threshold can work with. `C=1` has marginally better dev accuracy (0.607 vs 0.598, within noise); I chose `C=10` for usable confidence.
* I also tried four other character-analyser settings in an exploratory run (not saved as a report); all five variants scored 0.592-0.605 dev accuracy, within noise, so I stopped tuning rather than overfit 270 groups.
* Both choices were made on training-data cross-validation **before** the test set was ever evaluated.

## 7. FAQ answers (`faq.py` + `intent_faq_map.json`)

Because `hr_faq.csv` has no intent column, I wrote the mapping by reading all 30 answers and the training phrasings of every intent. It is **not** name-matching.

* Each FAQ-capable intent has ordered **rules**: `require_any` (a topic word the message must contain), `match_any` (at least one must hit), `veto_any` (blocks the rule), and one or more `faq_ids`. The matching rule with the most hits wins; ties go to the first-listed rule - fully deterministic.
* An answer is the **verbatim** `answer` text. A generic leave-policy question gets the three approved entitlement answers (FAQ-01, 02, 03) joined; nothing is invented.
* **Sub-questions the FAQ does not answer fall back.** Example: `pf_info` is recognised for "how do I activate UAN", but FAQ-13 only explains the PF *calculation*, so the bot refuses instead of reciting the wrong answer. `reports/faq_mapping_validation.csv` lists every training phrasing and the FAQ it gets (or "no approved answer"); I reviewed it by eye.
* **Guard rails found by probing, not by the test set:** "how is LOP calculated" was first answered with the PF formula (confidence 0.67). The PF rule now needs `pf`/`provident` in the message. "not my salary" used to get the pay-date answer; the pay-date rule now needs a when/date/day/payday/credit cue.
* **Intents with no FAQ answer at all (21 of 33):** the 6 conversational/out-of-scope ones (`greet`, `goodbye`, `thanks`, `chitchat`, `help`, `out_of_scope`), `apply_leave_howto`, and every intent that needs the employee's own or restricted data (`leave_balance`, `my_manager`, `salary_of_other`, `payslip_status`, ...). The 12 intents that do have rules (`hr_policy_leave`, `working_hours`, `wfh_policy`, `holiday_list`, `payslip_date`, `reimbursement_policy`, `notice_period_policy`, `probation_policy`, `pf_info`, `tax_declaration`, `insurance_benefits`, `who_to_contact`) answer some or all of their training phrasings - from 26/26 messages for `hr_policy_leave` and `wfh_policy` down to 0/26 for `holiday_list`, whose list requests are not in the FAQ (`reports/faq_coverage.md`).
* **8 of the 30 FAQ entries can never be returned**, because no intent in the data covers them: FAQ-08 overtime, FAQ-10 LOP, FAQ-11 bonus, FAQ-12 salary revision, FAQ-14 ESI, FAQ-15 professional tax, FAQ-27 bank details, FAQ-29 dress code.
* **Only approved contact details are shown.** The generic fallback contains none. The "no approved answer" reply for payroll-related intents (`payslip_status`, `payroll_flags`, `payslip_date`) appends FAQ-28's payroll contact verbatim. No HR-wide email or phone exists in the data, so none is invented.
* **FAQ self-check** (`reports/faq_question_self_check.csv`): the 30 official FAQ questions sent through the whole pipeline: **15 answered correctly, 0 answered with a different FAQ, 8 refused because no intent covers them, 7 refused because confidence was below the threshold** (e.g. "How many sick leaves do I get?" scored 0.614).

## 8. Confidence threshold and fallback

* **Threshold = 0.64**, stored in the artifact, overridable with `CORTEXHR_CHAT_THRESHOLD`.
* **Selection (training data only):** out-of-fold probabilities from the grouped cross-validation above. Rule: the *smallest* threshold at which at least 95% of the messages the bot would answer are the right intent, with at least 50 answered dev messages. At 0.64: 319 of 858 dev messages answered (37.2%), 95.3% of them correct, 15 wrong. (At the brief's 0.60 the figure is 93.0%, below the 95% target; at 0.70 it is 97.5% but only 33% are answered.) The full curve is `reports/threshold_selection_curve.csv`. **The official test set was not used to choose or tune it.**
* **What the bot does** (each case has a test): empty/whitespace message; the model raising an error or returning an invalid distribution; probability below the threshold; the model saying `out_of_scope`; a recognised intent with no approved answer.
* **Fallback text:** `I'm not sure I understood your question. Please rephrase it, or contact HR for assistance.` (uncertain / empty / error / out of scope) or `I don't have an approved answer for that question yet. Please contact HR for assistance.` (recognised intent, no approved FAQ answer).
* **The confidence is not a calibrated probability.** On the dev split the expected calibration error is 0.11 (`reports/dev_reliability_table.csv`): the 0.5-0.6 band is right 79% of the time (under-confident) while the 0.6-0.7 band is right only 71% of the time (a little over-confident). A high score never guarantees a correct answer, and a simple threshold cannot reliably detect every out-of-scope question (see the two confident test errors below).

## 9. Evaluation on `chatbot_test_set` (132 messages, run once on the frozen model)

| metric | value |
|---|---|
| **accuracy** | **0.803** (106 / 132) - 95% Wilson interval 0.727-0.862 because n is only 132 |
| **macro F1** | **0.800** |
| **weighted F1** | **0.800** |
| intents with F1 = 1.0 | 6 of 33: `apply_leave_howto`, `attendance_summary`, `leave_balance`, `leave_request_status`, `probation_policy`, `tax_declaration` |

Per-intent precision/recall/F1: `reports/classification_report.csv`. Lowest F1: `help` 0.33, `chitchat` 0.55, `team_members` 0.57, `thanks` 0.57, `who_to_contact` 0.67, `out_of_scope` 0.67, `notice_period_policy` 0.67, `holiday_list` 0.67. Each intent has only 4 test messages, so one message moves its recall by 25 points.

Confusion matrix: `reports/confusion_matrix.csv` and `reports/confusion_matrix.png`. Pair counts: `reports/confused_pairs.csv`.

**Effect of the threshold on the test set (information only - it was fixed beforehand and not changed):** 58 of 132 messages are above 0.64; 56 of those 58 are correct (96.6%). Of the 26 wrong predictions, **24 are below the threshold and would be refused**, but 50 *correct* predictions are also refused. The mean confidence of correct predictions is 0.647 against 0.293 for wrong ones.

### End-to-end behaviour of the chatbot on the same 132 messages (`end_to_end` in `reports/intent_evaluation.json`)

| outcome | messages |
|---|---|
| approved FAQ answer returned | **13** (9.8%) |
| ... of which for the wrong intent | **0** |
| fallback: confidence below 0.64 | 74 |
| fallback: intent recognised, no approved answer | 45 |

Only 48 of the 132 test messages belong to an intent that has any FAQ rules at all (12 intents x 4), so 48 is the most the bot could ever answer on this test set; it answered 13 of them. I read all 13 replies: **11 are clearly appropriate; 2 are only partly right** - TST-047 "can I leave earlier than the notice period" received the notice-period *lengths* (FAQ-18) although the buy-out answer (FAQ-19) fits better, and TST-024 "calendar of holidays" received a pointer to the January calendar, not a list. The test set has no label for "was this the right sub-answer", so this is my manual judgement. I did not change the rules after seeing these (that would be tuning on the test set); a better FAQ-19 rule is listed under improvements.

### Error analysis (all 26 misclassified messages: `reports/misclassified_examples.csv`, with true intent, predicted intent and confidence)

* **Two confident mistakes (the dangerous kind):** "my team name" -> `team_members` (0.857, true `my_department`) and "how many days notice do I need to serve" -> `wfh_policy` (0.702, true `notice_period_policy`). The second matches the training phrase "how many days do I need to be in office". The threshold cannot stop these (both are above 0.64). What stopped them in the *chatbot* is the FAQ layer: neither message matches a rule of the wrongly predicted intent (`team_members` has no FAQ; the `wfh_policy` rule needs a work-from-home/remote/office word), so both ended as "no approved answer" instead of a wrong HR answer (see "End-to-end" below). That protection is a property of these two particular messages, not a guarantee.
* **Out-of-vocabulary paraphrases:** "yo", "cheers", "appreciate it", "who works under me", "how do I quit the company", "how to get my travel money back", "will it rain tomorrow", "do we get Holi off" - wording that never appears in training. Most of these land at confidence 0.06-0.24.
* **Small intents with fuzzy borders:** `help`/`chitchat`/`greet`, and the "team" intents.

### The two most confused intent pairs

The test set has only 4 messages per intent, so after the first pair the ranking is a **tie** (many pairs have exactly 1 confusion). I therefore also used the dev split (858 messages, `reports/dev_confused_pairs.csv`) for the second pair.

1. **`help` <-> `chitchat`** - the only pair with more than one test confusion (2: "what can this bot do" and "list your features" were both predicted `chitchat`; dev split: 12 `help`->`chitchat`). Cause: the labels overlap. In the training data "what can you do" is labelled `chitchat` while "what can I ask you" and "what do you support" are `help`. `help` has just 5 distinct phrasings.
   *Recommended data changes:* decide one rule, capability/feature/commands questions -> `help`, identity and pleasantries ("what is your name", "how are you") -> `chitchat`; relabel the training example "what can you do" to `help`; add at least 15 new, genuinely different `help` phrasings (what can this bot do, which topics are covered, how do I use you, list commands...) and at least 10 `chitchat` identity phrasings (are you a person, who made you, what should I call you...) so the two vocabularies separate. Collect new phrasings; do not copy the test sentences.
2. **`team_members` <-> `my_manager` (with `my_department`)** - 1 test confusion in this pair ("who works under me" -> `my_manager`) and the same cluster accounts for three of the 26 test errors; dev split: 11 `my_manager`->`team_members`. (The single largest dev confusion, `unusual_attendance`->`attendance_summary` with 20, did not occur on the test set.)
   *Recommended data changes:* each of these three intents has only 4-5 distinct phrasings built on the same words ("my", "team", "report"). Add contrastive pairs about **direction**: people above me (who is my boss/lead/line manager) vs people below me (my reportees, who works under me, my subordinates) vs peers (my teammates, people in my team), and "which team/department am I in, what is my team called" for `my_department`. Add at least 15 distinct phrasings per intent, including informal ones.

**Other data recommendations:** (a) the median intent has only 6 distinct phrasings - aim for 30+ real paraphrases per intent rather than more filler/typo copies of the same sentence; (b) add colloquial/short forms ("yo", "cheers", "thx", "bye then"); (c) add the official FAQ question wordings as training examples (7 reachable FAQ questions currently miss the threshold) - but then build a **new, untouched test set**, because this one would no longer be independent; (d) add intents for the 8 unreachable FAQs (overtime, LOP, bonus, salary revision, ESI, professional tax, bank-details update, dress code); (e) add a few negative "near-HR but unsupported" examples to `out_of_scope` (the model currently sees only 7 phrasings of it); (f) write down labelling rules for borderline intents; (g) rule improvements I noticed but did not apply because they came from reading test messages: the notice-period FAQ-19 (early release / buy-out) rule should also trigger on wording such as "leave earlier", and the holiday intent needs a real holiday list in `hr_faq`. I did **not** apply any of these: doing so after reading the test errors and then re-scoring on the same test set would inflate the result.

## 10. The endpoint: `POST /v1/chat`

Request: `{"message": "<text up to 1000 characters>"}`. Extra fields, a missing field, a non-string and over-long text give **422** (same error body style as Task 8: `loc`/`type`/`msg`, the submitted value is never echoed). Empty or whitespace-only text is valid input and returns the fallback with HTTP 200. If the model artifact could not be loaded at startup, `/v1/chat` returns **503** and every Task 8 endpoint keeps working. No authentication (Task 8 has none).

Response: exactly `intent` (string or null), `confidence` (0-1 or null), `reply`, `fallback`. Real outputs (full list in `reports/example_conversations.md` and `reports/task9_api_smoke_output.txt`):

```json
{"intent":"notice_period_policy","confidence":0.9458,"reply":"30 days for L1-L2, 60 days for L3, 90 days for L4 and above.","fallback":false}
{"intent":"payslip_date","confidence":0.8913,"reply":"On the first working day of the following month.","fallback":false}   // "whn is salary credited yaar"
{"intent":null,"confidence":0.1836,"reply":"I'm not sure I understood your question. Please rephrase it, or contact HR for assistance.","fallback":true}   // "what is the dress code"
{"intent":null,"confidence":null,"reply":"I'm not sure I understood your question. Please rephrase it, or contact HR for assistance.","fallback":true}   // "   "
```

The model, FAQ and rules are loaded **once** in the app's lifespan handler; a request only calls `predict_proba`. Responses never contain stack traces, paths, training text or the FAQ table; logs record the reason code and exception *type* only, never message text.

Environment variables (all optional): `CORTEXHR_TASK9_ROOT` (folder containing `src/cortexhr_task9`; this checkout's canonical path is `<repo>/phase-2/task09/cortexhr_task9/task 9`), `CORTEXHR_INTENT_MODEL`, `CORTEXHR_FAQ_FILE`, `CORTEXHR_CHAT_THRESHOLD`.

## 11. Run it locally

Use the same environment as Task 8 (`phase-2/task08/task 8/requirements.txt`; **Task 9 adds no runtime dependency**). `matplotlib` is optional and only draws `confusion_matrix.png`. The artifact records the scikit-learn / NumPy / SciPy / joblib versions it was built with and the service **refuses to load it on a mismatch**, so build it in the environment you serve from. The model in `models/` is git-ignored and must be rebuilt locally. `joblib` uses pickle: only load artifacts you built yourself.

Windows PowerShell (not run on Windows - see limitations):

```powershell
Set-Location 'C:\path\to\CortexHR-Phase-2-3'
python -m pip install -r '.\phase-2\task08\task 8\requirements.txt'
python '.\phase-2\task09\cortexhr_task9\task 9\scripts\train_intent_model.py'    # builds local model + dev reports
python '.\phase-2\task09\cortexhr_task9\task 9\scripts\evaluate_intent_model.py' # evaluates on chatbot_test_set (read-only)
python -m pytest '.\phase-2\task09\cortexhr_task9\task 9\tests' -v

# start the Task 8 service (it loads the chat model once at startup)
Set-Location '.\phase-2\task08\task 8'
$env:CORTEXHR_TASK9_ROOT = (Resolve-Path '..\..\phase-2\task09\cortexhr_task9\task 9').Path
$env:CORTEXHR_WORKBOOK = 'C:\path\to\02_CortexHR_AI_ML_Dataset.xlsx'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Test it from a second PowerShell window:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat -ContentType 'application/json' `
  -Body (@{ message = 'How long is the notice period?' } | ConvertTo-Json)

curl.exe -sS -X POST http://127.0.0.1:8000/v1/chat -H "Content-Type: application/json" -d "{\"message\": \"whn is salary credited yaar\"}"
```

Linux/macOS: use the same Python commands with platform-appropriate paths; `curl -X POST http://127.0.0.1:8000/v1/chat -H 'Content-Type: application/json' -d '{"message": "What is the notice period?"}'`. The Task 5-7 source is in this checkout, so no separate `CORTEXHR_TASKS_ROOT` is required.

## 12. How I verified it works

* `python -m pytest -v` in `task 9`: **152 passed, 0 failed, 0 skipped** (`reports/task9_test_output.txt`). By file: text cleaning 19, intent model 14, FAQ 53, chat service 41, API 25. They cover all 15 required checks (artifact load, FAQ mapping, known questions, approved answers, low confidence, empty / whitespace / missing field, unsupported questions, no-FAQ intent, confidence range, response schema, Task 8 routes, no retraining or reloading per request, no leaked error details) and refusal of tampered artifacts (wrong scikit-learn/NumPy version, old cleaning version, bad format, corrupted or incomplete file).
* **Mutation check:** I deliberately broke five behaviours (removed the PF guard, changed `>=` to `>` at the threshold, stopped catching inference errors, made the typo map rewrite `form`, made training import the test loader). Each was caught by a failing test; the suite was restored and passed again.
* **Live HTTP check:** I started uvicorn and sent real requests (`reports/task9_api_smoke_output.txt`): 200 answers and fallbacks, 422 for `{}` and a number, and `/v1/health` still 200.
* **Reproducibility:** training is deterministic (a test retrains and compares probabilities and threshold; the evaluation numbers were identical across re-runs).

### Verification limits from the original Task 9 development run

* The original isolated Task 9 assessment did not have the full Task 5-8 source or reference workbook and used temporary Task 8 stubs for route checks; it did **not** verify real anomaly scoring. In this combined repository Tasks 5-8 are present. Follow their READMEs and run their tests and workbook-backed checks locally before relying on the integrated service.
* The original assessment ran on Linux with Python 3.12.3; its Windows / PowerShell commands were not tested there. Rebuild the model artifact in the actual serving environment rather than reusing a checked-out binary.

## 13. Known limitations

1. **Accuracy 80% on 132 test messages** (interval 73-86%). On the training dev split, which holds out whole phrasings, it is 60%. Expect worse on wording unlike the training data.
2. **Cautious by design, and not very helpful yet:** at 0.64 the bot refuses 56% of the test messages (including 50 it classified correctly), and in total it returns an FAQ answer for only 13 of 132 (9.8%). Natural FAQ wordings such as "how many sick leaves do I get?" (0.614) are refused.
3. **Confidence is uncalibrated** (ECE 0.11). The threshold was chosen on 858 dev messages (15 wrong answers at that threshold); on the test set 2 of the 58 messages above it were wrong intents.
4. **The classifier does not understand negation or meaning**; it matches words and character patterns. It can be wrong and confident (e.g. "how many days notice do I need to serve" -> `wfh_policy`).
5. **Only 22 of 30 FAQ entries are reachable**, and many intents only partially covered. Conversational intents (`hello`, `thanks`, `bye`) have no approved reply text in `hr_faq`, so they currently get the "no approved answer" message; add approved conversational replies if you want greetings to work.
6. **Questions that need the employee's own data** (leave balance, payslip, manager) are recognised but not answered: that is Task 10 (entities, memory, permissions). The `entities` columns in the data are not used here.
7. **Out-of-scope detection** relies on the 7-phrasing `out_of_scope` class plus low confidence; it is not a reliable scope detector.
8. Unrelated words in a rule-matched sentence can still trigger a rule (the rules are word-based). Rules are English only.
9. **Not production-ready.** It has no authentication, rate limiting, monitoring, human review of answers, privacy review of logged intents, or evaluation on real employee messages.

## 14. Evidence files (`reports/`)

| file | what it shows |
|---|---|
| `intent_evaluation.json`, `classification_report.csv` | test accuracy, macro/weighted F1, per-intent precision/recall/F1, overlap check, end-to-end chatbot outcomes |
| `confusion_matrix.csv`, `confusion_matrix.png`, `confused_pairs.csv` | confusion matrix and ranked confusion pairs |
| `misclassified_examples.csv` | all 26 wrong test messages with true/predicted intent and confidence |
| `example_conversations.md` | three real conversations (supported, informal, unsupported) plus other real attempts |
| `model_config_comparison_dev.csv`, `threshold_selection_curve.csv`, `dev_reliability_table.csv`, `dev_confused_pairs.csv`, `dev_per_intent_accuracy.csv` | dev-split (training-only) evidence for model and threshold choices |
| `faq_mapping_validation.csv`, `faq_coverage.md`, `faq_question_self_check.csv` | what each phrasing / FAQ gets |
| `task9_test_output.txt`, `task9_evaluation_output.txt`, `task9_api_smoke_output.txt` | raw outputs of the test run, evaluation, and live HTTP calls |

## 15. Git and data hygiene

The artifact (`models/*.joblib`, `*.metadata.json`) is git-ignored like Task 8's models. The three starter CSVs are included under the canonical Task 9 `data/` directory. Confirm their redistribution is approved before repackaging this repository; never add employee-level inputs or private data.
