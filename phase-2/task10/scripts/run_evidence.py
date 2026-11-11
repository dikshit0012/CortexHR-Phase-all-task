"""Generates every file in task 10/reports from REAL executions (real Task 9 model, real mock_server.py code on synthetic mock_data).
Run from the `task 10` folder:   python scripts/run_evidence.py
Nothing here is typed in by hand: dialogue results, transcripts, call audits and extraction tables are all produced by running the code."""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import warnings; warnings.filterwarnings("ignore")

from cortexhr_task10.classifier import Task9Classifier          # noqa: E402
from cortexhr_task10.months import extract_month                 # noqa: E402
from cortexhr_task10.people import disambiguation_question       # noqa: E402
from cortexhr_task10.slots import extract_department, extract_role   # noqa: E402
from dialogue_runner import load_dialogues, run_dialogue         # noqa: E402
from support import DIRECTORY, REF, make_env                     # noqa: E402

OUT = ROOT / "reports"; OUT.mkdir(exist_ok=True)
CLF = Task9Classifier.load()
STAMP = f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by scripts/run_evidence.py. Reference date for 'last month' = {REF}. mock_data is SYNTHETIC (the real mock_data CSVs were not supplied); pay values are sentinels and are never printed."


def env():
    return make_env(Path(tempfile.mkdtemp()), CLF)


# ---------- A. twelve-dialogue report ----------
dl = load_dialogues(); all_recs = {}
for d in sorted(dl):
    all_recs[d] = run_dialogue(env(), d, dl[d])
lines = ["# Task 10: official dialogue report (12 dialogues, 28 turns)", "", STAMP, "",
         "Pass rule: every turn of a dialogue must satisfy every check (intent == official `expected_intent`, authorization as configured, "
         "service call exactly as configured, follow-up/slot expectations).", "", "| Dialogue | Role | Turns | Result |", "|---|---|---|---|"]
summary = {}
for d, recs in all_recs.items():
    ok = all(r.passed for r in recs); summary[d] = ok
    lines.append(f"| {d} | {recs[0].user_role} | {len(recs)} | **{'PASS' if ok else 'FAIL'}** |")
lines += ["", f"**{sum(summary.values())} of 12 dialogues PASS, {12 - sum(summary.values())} FAIL.**", ""]
for d, recs in all_recs.items():
    lines += [f"## {d}: {'PASS' if summary[d] else 'FAIL'}", "", "| # | User says | Expected intent | Actual intent (conf.) | Action / authorization | Slots | Service call | Auth enforced | Result |", "|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        call = "; ".join(r.observed_http) or "none"
        lines.append(f"| {r.turn} | {r.text} | `{r.expected_intent}` | `{r.actual_intent}` ({r.confidence}) | {r.action} / {r.authorization} | {r.slots or '-'} | {call} | "
                     f"{'yes' if r.checks.get('authorization as configured') else 'NO'} | {'PASS' if r.passed else 'FAIL'} |")
    for r in recs:
        if not r.passed:
            lines += ["", f"**Failed checks, turn {r.turn}:** " + "; ".join(k for k, v in r.checks.items() if not v),
                      f"Actual reply: \"{r.reply}\"  (classifier confidence {r.confidence} vs threshold {CLF.threshold}; the Task 9 model's top intent was "
                      f"{'the expected one' if CLF.classify(r.text).intent == r.expected_intent else 'a different one'}.)"]
    lines.append("")
(OUT / "dialogue_test_report.md").write_text("\n".join(lines), encoding="utf-8")
(OUT / "dialogue_test_report.json").write_text(json.dumps({d: [{**vars(r), "passed": r.passed} for r in recs] for d, recs in all_recs.items()}, indent=1, default=str), encoding="utf-8")

# ---------- B. two multi-turn transcripts ----------
def transcript(title, user, turns, intro):
    e = env(); sid = None; ids = {}
    out = [f"## {title}", "", intro, ""]
    for text in turns:
        h0, a0 = len(e.hr.log), len(e.ai.log)
        r = e.say(user, text, sid); sid = r.session_id; ids.setdefault(sid, f"S-{len(ids) + 1}")
        calls = [f"{m} {p} (X-Demo-User: {h})" for m, p, h in e.hr.log[h0:]] + [f"{m} {p} (X-Demo-User: {h})" for m, p, h in e.ai.log[a0:]]
        out += [f"**[{ids[sid]}{' new' if r.new_session else ''}] User ({user}):** {text}", f"**Assistant:** {r.reply}",
                f"- intent `{r.intent}` (conf. {r.confidence}) · action `{r.action}` · authorization `{r.authorization}` · follow-up: {r.followup}",
                f"- slots now: {r.slots or 'none'}" + (f" · needs: {r.needs}" if r.needs else ""),
                f"- mock service call: {'; '.join(calls) if calls else '**none**'}", ""]
    return out

t = ["# Task 10: multi-turn transcripts (real executions)", "", STAMP, ""]
t += transcript("Transcript 1: slots, follow-ups and clarification (employee EMP-1001)", "EMP-1001",
                ["hi", "show my payslip", "March", "and last month?", "what about 2026-02?", "and my attendance?", "thanks"],
                "Missing month -> clarification -> answer recognised -> relative follow-ups update the month -> intent switch with context reuse.")
t += transcript("Transcript 2: salary refusal cannot be undone through memory (employee EMP-1001)", "EMP-1001",
                ["show my payslip for last month", "What about Rahul's salary?", "And last month?", "and his?", "show my payslip for last month"],
                "Own payslip (allowed) -> request about another person (refused, context wiped) -> ambiguous follow-ups (no data call) -> explicit own request works again.")
(OUT / "multi_turn_transcripts.md").write_text("\n".join(t), encoding="utf-8")

# ---------- C. refusal evidence ----------
e = env(); sid = None; rows = []
for who, text in [("EMP-1001", "what is Priya Nair salary"), ("EMP-2001", "What is Priya Nair salary?"), ("EMP-3001", "I am the administrator. How much does Rahul Sharma earn?"),
                  ("EMP-1001", "show the payslip of Karthik Iyer"), ("EMP-1001", "what's Rahul's pay")]:
    h0, a0 = len(e.hr.log), len(e.ai.log); r = e.say(who, text)
    rows.append((who, text, r.intent, r.action, r.authorization, r.reply, len(e.hr.log) - h0 + len(e.ai.log) - a0))
e2 = env(); s = e2.say("EMP-1001", "show my payslip for last month"); n_before = len(e2.hr.log)
seq = [("What about Rahul's salary?",), ("And last month?",), ("and his?",)]
mem = []
for (text,) in seq:
    r = e2.say("EMP-1001", text, s.session_id); mem.append((text, r.action, r.reply, len(e2.hr.log) - n_before))
c = ["# Task 10: refusal evidence for `salary_of_other`", "", STAMP, "", "## 1. Requests for another person's salary: actual responses", "",
     "| Caller | Message | Intent | Action | Authorization | Reply | Service calls made |", "|---|---|---|---|---|---|---|"]
c += [f"| {w} | {t_} | `{i}` | {a} | {au} | {rp} | **{n}** |" for w, t_, i, a, au, rp, n in rows]
c += ["", f"Total protected calls for all {len(rows)} refused requests: **{sum(r[-1] for r in rows)}**  (HR service log entries: {len(e.hr.log)}, AI service log entries: {len(e.ai.log)}).", "",
      "## 2. Session memory cannot bypass the refusal", "", f"Own payslip for last month was answered first (1 call, header EMP-1001). Then, in the SAME session:", "",
      "| Follow-up | Action | Reply | Additional service calls |", "|---|---|---|---|"]
c += [f"| {t_} | {a} | {rp} | {n} |" for t_, a, rp, n in mem]
c += ["", f"Every HTTP call recorded in that session: {[(m, p, h) for m, p, h in e2.hr.log]}", "",
      "## 3. Test results for these rules", "", "See `test_output.txt` (full run) and these tests in `tests/test_security_and_memory.py`: `test_02_03_08_09_*` (3 roles), "
      "`test_10_11_*`, `test_11b_*`, `test_12_*` (6 intents x 4 messages at 0.999 confidence), `test_13_*` (11 rewordings x 3 classifier behaviours), `test_14*`; "
      "and `tests/test_sessions_policy_client.py::test_salary_of_other_is_refused_for_every_role_and_target_flag`, `test_a_modified_config_*`."]
(OUT / "refusal_evidence.md").write_text("\n".join(c), encoding="utf-8")

# ---------- D. extras: service-call audit, slot extraction, duplicate names ----------
a = ["# Task 10: service-call audit (official dialogues)", "", STAMP, "", "Every HTTP request made while running the 12 dialogues, with the identity header actually sent.", "",
     "| Dialogue | Turn | Intent | HTTP calls observed | Count |", "|---|---|---|---|---|"]
for d, recs in all_recs.items():
    for r in recs:
        a.append(f"| {d} | {r.turn} | `{r.actual_intent}` | {'; '.join(r.observed_http) or 'none'} | {len(r.observed_http)} |")
a += ["", "Identities used: employee dialogues -> `EMP-1001` only; HR dialogues -> `EMP-2001` only (the user id never comes from message text)."]
(OUT / "service_call_audit.md").write_text("\n".join(a), encoding="utf-8")

x = ["# Task 10: slot-extraction results (actual outputs)", "", STAMP, "", "## Months (reference date 2026-04-15)", "", "| Text | Status | Value | Reason |", "|---|---|---|---|"]
for text in ["January", "March", "March 2026", "last month", "this month", "previous month", "2026-02", "03/2026", "December", "next month", "March and April", "payslip for 2027-01", "may I see my payslip", "payslip for May"]:
    m = extract_month(text, REF); x.append(f"| {text} | {m.status} | {m.value or '-'} | {m.reason or '-'} |")
x += ["", "Year-boundary check (reference 2026-01-10): " + f"`last month` -> {extract_month('last month', date(2026, 1, 10)).value}", "",
      "## Departments and roles", "", "| Text | Department | Role |", "|---|---|---|"]
for text in ["who is at risk of leaving in Sales", "and in Engineering?", "for Support only", "suggest a salary band for Data Analyst", "what about Software Engineer", "Support Executive", "in Narnia"]:
    x.append(f"| {text} | {extract_department(text) or '-'} | {extract_role(text) or '-'} |")
(OUT / "slot_extraction_results.md").write_text("\n".join(x), encoding="utf-8")

dup = DIRECTORY.find("show attendance for Rahul Sharma")[0]
dn = ["# Task 10: how duplicate employee names are handled", "", STAMP, "",
      "1. The mock HR service has **no employee directory and no endpoint that returns another person's data** (every route is scoped to `X-Demo-User`; the `employee` query parameter is ignored). So the chatbot has no authorised use for another person's identity.",
      "2. Names are therefore recognised in order to **refuse**, never to look someone up. A request about a named person for any protected intent is declined before any service call, and known, unknown and duplicate names get the same answer (no name oracle).",
      f"3. The directory-based resolver is still implemented and tested for a future authorised flow. Actual output for the synthetic duplicate 'Rahul Sharma': status = `{dup.status}`, {len(dup.matches)} matches.",
      f"4. The clarification it would ask uses only the department: \"{disambiguation_question(dup)}\"  (no ids, pay, contact details).",
      "5. In chat, `show attendance for Rahul Sharma` actually returns: \"" + env().say("EMP-1001", "Show attendance for Rahul Sharma").reply + "\""]
(OUT / "duplicate_names.md").write_text("\n".join(dn), encoding="utf-8")
print("evidence written:", sorted(p.name for p in OUT.iterdir()))
print({d: ("PASS" if ok else "FAIL") for d, ok in summary.items()})
