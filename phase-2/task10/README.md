# CortexHR Phase 2 · Task 10: Make the chatbot smart (slots, memory, permissions)

## 1. Objective
Turn the Task 9 intent classifier into a conversational HR assistant that **extracts months**, **remembers context across messages**, **calls the mock HR service** and **enforces role-based access**, with one rule above all: *nobody can obtain another person's salary by rewording, by manipulating the conversation, or by exploiting session memory.*

## 2. Files
```
phase-2/task10/
  data/chatbot_dialogues.csv, chatbot_responses.csv   starter dialogue and policy data (included)
  starter/mock_services/mock_server.py                mock service code; its mock_data/ CSV files are not included
  src/cortexhr_phase-2/task10/
    months.py        month extraction -> 'YYYY-MM'          slots.py     department / role / follow-up cues / "slot-only" test
    people.py        employee directory, partial & duplicate names, safe disambiguation question
    sensitive.py     pre-authorization detection of "about someone else" and "about pay"
    identity.py      X-Demo-User -> Identity(employee_id, role) from a server-side table
    policy.py        loads chatbot_responses.csv; the ONE authorize() function
    sessions.py      in-memory session store          service_client.py   HR + AI HTTP clients (no dialogue logic)
    render.py        service data -> official response_template       dialogue.py   orchestration
    classifier.py    adapter over the Task 9 model      api.py   POST /v1/chat/converse      config.py   settings
  tests/             synthetic and service-contract tests          scripts/run_evidence.py   evidence runner
  reports/           retained evidence from an earlier run; some outputs need excluded inputs
phase-2/task08/task 8/app/converse.py, phase-2/task08/task 8/app/main.py    Task 10 route integration
phase-2/task09/cortexhr_task9/task 9/                             canonical Task 9 model integration
```
`POST /v1/chat` (Task 9) is unchanged; Task 10 adds `POST /v1/chat/converse` next to it.

## Setup prerequisites and tests

The source and starter dialogue/policy CSVs are included, but a clean checkout is **not runnable end to end**. Obtain the mock service CSVs `employees.csv`, `leave_balances.csv`, `leave_requests.csv`, `payslips.csv`, `holidays.csv`, `attendance_monthly.csv`, and `demo_users.csv` through an approved channel. The mock service loads them from `phase-2\task10\starter\mock_services\mock_data\` at startup. They are intentionally excluded; do not generate or commit employee-level data or demo credentials.

Install the Task 8 environment and run the Task 10 tests from the repository root:

```powershell
python -m pip install -r 'phase-2\task08\task 8\requirements.txt'
python -m pytest phase-2\task10\tests
```

The mock server is a demo only. `X-Demo-User` is spoofable and must never be treated as authentication. HR-only AI routes are external dependencies; this repository does not provide their service implementation.

Before starting the API, build Task 8's anomaly artifacts and train the canonical Task 9 model using their setup guides. Use the same environment and compatible model-library versions for both services. The Task 8 API itself requires the approved reference workbook.

## 3. Start the mock HR service
```powershell
Set-Location 'C:\path\to\CortexHR-Phase-2-3\phase-2\task10\starter\mock_services'
python -m uvicorn mock_server:app --host 127.0.0.1 --port 9000
```
## 4. Start the CortexHR API
```powershell
Set-Location 'C:\path\to\CortexHR-Phase-2-3\phase-2\task08\task 8'
$env:CORTEXHR_HR_BASE_URL       = "http://127.0.0.1:9000"
$env:CORTEXHR_DEMO_USERS_CSV    = "C:\path\to\CortexHR-Phase-2-3\phase-2\task10\starter\mock_services\mock_data\demo_users.csv"
$env:CORTEXHR_REFERENCE_DATE    = "2026-04-15"     # optional: pin what "last month" means; default = today
$env:CORTEXHR_AI_BASE_URL       = "http://approved-ai-service:8000"   # optional external HR-only routes; see §18
$env:CORTEXHR_TASK9_ROOT        = "C:\path\to\CortexHR-Phase-2-3\phase-2\task09\cortexhr_task9\task 9"
$env:CORTEXHR_WORKBOOK          = "C:\approved-data\02_CortexHR_AI_ML_Dataset.xlsx"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
## 5. Send a chat request
```powershell
$h = @{ "X-Demo-User" = "EMP-0188" }
$r = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/converse -Headers $h -ContentType "application/json" -Body '{"message":"show my payslip for last month"}'
$r.reply; $r.session_id
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/converse -Headers $h -ContentType "application/json" -Body (@{message="and the month before?"; session_id=$r.session_id} | ConvertTo-Json)
```
Response: `session_id, new_session, intent, confidence, reply, fallback, action (static|answered|clarify|refused|denied|failed|fallback), needs, slots, followup`.

## 6. Session IDs
Server-issued random ids (`secrets.token_urlsafe(24)`). A client may send one back (16-64 chars `[A-Za-z0-9_-]`; employee ids such as `EMP-0188` and e-mail addresses cannot match). State is keyed by **(authenticated employee, session id)**: another user sending the same id gets a brand-new empty session and a new id, so ids are never proof of anything and cannot read someone else's memory. Unknown/expired ids (30 min idle) silently start a fresh session. Stored: last intent, month/department/role, one pending question. **Never stored:** message text, names, pay values.

## 7. Month and employee extraction
*Months*: `March`, `March 2026`, `2026-03`, `03/2026`, `last/previous/prior month`, `this/current month` → `YYYY-MM`. A bare month name means the **most recent such month not after the reference month** (rule is explicit, and the resolved `YYYY-MM` is always shown in the reply, e.g. "Your payslip for 2026-03…"). `next month`, a future month, or two different months → the bot asks again. "may" counts as a month only in a month-like context ("payslip for May", not "may I see…").
*Employees*: the mock service has **no directory and no route returning anyone else's data**; every endpoint serves only the caller. So names are recognised to **refuse**, never to look up (`sensitive.py`: capitalised names, possessives, "salary of x", "how much does x earn", "what does x make", pronouns, "colleague"…, plus an optional `EmployeeDirectory`). The employee slot is always the authenticated caller.

## 8. Follow-ups
After an answered data question the session remembers intent + slots. A message that is **only** a slot value (`and last month?`, `what about April?`, `and in Engineering?`, `for Support only`, `and sick leave?`), or a detail cue (`show details`, `why is it high`), re-uses the previous intent with the new slot; unchanged slots are reused only when the message carries a follow-up cue (`and…`, `what about…`). **Authorization runs again on every follow-up.** A leftover unknown word (e.g. a name) disqualifies a message from being a follow-up.

## 9. Missing slots
Required slots (derived from the official placeholders): `month` for payslip_status / attendance_summary / payroll_flags, `department` for attrition_risk_query, `role` for salary_band_query / resume_score_request. If missing the bot asks ("Which month would you like?", "Which department do you mean?", "Which role do you mean?"), remembers the pending question, validates the next message as the answer, and **makes no service call until it is filled**. Changing topic drops the pending question.

## 10. Duplicate employee names
`people.py` can resolve full/partial/duplicate names and ask "I found more than one employee with that name…" listing **only departments**. In chat this is never used to reveal anything: since no authorised flow involves another person, the bot declines the request and answers known, unknown and duplicate names identically (no name oracle). See `reports/duplicate_names.md`.

## 11. `X-Demo-User`
The header is the **only** source of identity. The role comes from a server-side table (`demo_users.csv`), never from the message, a slot or the session. No header = anonymous (public intents only); malformed/unknown header = HTTP 401 with no echo. Every service call sends exactly the authenticated id as `X-Demo-User`; the client has no parameter to pass another. **This is a demo mechanism, not authentication:** anyone who can reach the API can claim any demo user. Production needs a verified token.

## 12. Roles and permissions
`policy.authorize(identity, intent, policies, target_is_self)` is the single decision point, driven by `chatbot_responses.csv` (`requires_login`, `allowed_roles`). Order: `salary_of_other` → deny; unknown intent → deny; public intent → allow; other-person target → deny; anonymous → login required; role not in {EMPLOYEE, HR, ADMIN} → deny; role not allowed → deny. It never sees classifier confidence, message text or session data, and it runs on every turn before slot filling and before any call.

## 13. Why `salary_of_other` is always refused
It is the official intent whose configured action is `REFUSE`. It is refused (a) by a text check *before* the classifier result matters (pay term + reference to someone else), (b) when the classifier says so at any role, (c) by `authorize()` **by name and by the REFUSE marker**, so editing the CSV cannot enable it, and (d) structurally: the service client cannot name another employee. A refusal also **wipes session context**, so `and last month?` afterwards cannot revive anything. Evidence: `reports/refusal_evidence.md`.

## 14. Service-client errors
Timeouts (2 s connect / 5 s read), connection errors, 401/403, 404, other 4xx/5xx, invalid JSON and wrong shapes all become a short `ServiceError(kind)` (no bodies, no payroll values, no URLs with data in logs). The user sees the official `on_failure` text; a 404 for a month says "I couldn't find any data for 2026-01". The month is validated (`^\d{4}-(0[1-9]|1[0-2])$`) before it is placed in a URL.

## 15. Run the 12 official dialogue tests
```powershell
Set-Location 'C:\path\to\CortexHR-Phase-2-3'
python -m pytest phase-2\task10\tests\test_official_dialogues.py -v
```
## 16. Run security and regression tests
```powershell
Set-Location 'C:\path\to\CortexHR-Phase-2-3'
python -m pytest phase-2\task10\tests -v
python -m pytest phase-2\task05\tests
python -m pytest 'phase-2\task06\task 6\tests'
python -m pytest 'phase-2\task07\task 7\tests'
python -m pytest 'phase-2\task09\cortexhr_task9\task 9\tests'
```
## 17. Evidence (`task 10/reports/`)
`dialogue_test_report.md/.json` (12 dialogues) · `multi_turn_transcripts.md` (2 transcripts) · `refusal_evidence.md` · `service_call_audit.md` · `slot_extraction_results.md` · `duplicate_names.md` · `test_output.txt` · `regression_task5_to_9_output.txt`. Regenerate with `python scripts/run_evidence.py`.

## 18. Known limitations and next steps
* **DLG-04 fails (11/12).** Turn 2 "any unusual days?" scores 0.5874 for the correct intent, under Task 9's 0.64 threshold, so the bot declines to act. Options: add training examples in Task 9, or an explicit "confirm?" step for 0.5-0.64. I did **not** lower the threshold. (`hi` also scores 0.06 on the model; an exact-match greeting rule, used only when the model is unsure, covers it.)
* **`mock_data/*.csv` were not supplied.** Tests run the real `mock_server.py` on **synthetic** data. Only columns the server code itself reads are certain; the rest (`cl/sl/el`, `present/scheduled/late`, `manager_name`, `department_name`, `status`, holiday `date/name`) are accepted under several likely names in `render.py` and are **unverified**. `my_manager` falls back to the manager's id if no name column exists.
* **`/ai/v1/...` routes are not served by `mock_server.py`** and no contract was provided, so HR-only intents (attrition, salary band, headcount, payroll flags, resume score) and `unusual_attendance` use **assumed** request bodies/response keys, tested against a stand-in. Without `CORTEXHR_AI_BASE_URL` they answer the official failure text. The Task 8 anomaly routes have different paths (`/v1/anomaly/...`) and need full records.
* **Task 8:** the real app code is now in place and exercised with a **stub scoring service** (`tests/test_task8_integration.py`, 5 tests: health, models, anomaly endpoints, Task 9 chat, converse, fail-closed behaviour). **Not run:** Task 8's own test-suite (not supplied) and loading the real model artifacts (they need `02_CortexHR_AI_ML_Dataset.xlsx` and Python 3.14.6 / pandas 3.1.0; I had neither). Treat 'anomaly scoring with real models' as NOT VERIFIED here.
* In-memory sessions are lost on restart and are **not shared between worker processes**; use one worker or add a shared store (Redis) before scaling.
* Detection of other people is heuristic. A lowercase unknown name with no cue ("priya nair payslip") can be classified as an own-payslip request; it then returns **the caller's own** payslip, because the service can only return the caller's data, but the intent is misread. Supplying an `EmployeeDirectory` narrows this.
* The demo header is spoofable (§11). Windows commands above were not run.
