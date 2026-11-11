# Task 10: multi-turn transcripts (real executions)

Generated 2026-10-30 03:14 UTC by scripts/run_evidence.py. Reference date for 'last month' = 2026-04-15. mock_data is SYNTHETIC (the real mock_data CSVs were not supplied); pay values are sentinels and are never printed.

## Transcript 1: slots, follow-ups and clarification (employee EMP-1001)

Missing month -> clarification -> answer recognised -> relative follow-ups update the month -> intent switch with context reuse.

**[S-1 new] User (EMP-1001):** hi
**Assistant:** Hello! I am the Zenith Nova HR assistant. Ask me about leave, payslips, attendance or policies.
- intent `greet` (conf. 0.0634) · action `static` · authorization `ok` · follow-up: False
- slots now: none
- mock service call: **none**

**[S-1] User (EMP-1001):** show my payslip
**Assistant:** Which month would you like? (for example "March 2026" or "last month")
- intent `payslip_status` (conf. 0.9819) · action `clarify` · authorization `ok` · follow-up: False
- slots now: none · needs: month
- mock service call: **none**

**[S-1] User (EMP-1001):** March
**Assistant:** Your payslip for 2026-03 is ready. You can download it from the Payslips screen.
- intent `payslip_status` (conf. 0.3187) · action `answered` · authorization `ok` · follow-up: True
- slots now: {'month': '2026-03'}
- mock service call: GET /api/v1/payslips/me/2026-03 (X-Demo-User: EMP-1001)

**[S-1] User (EMP-1001):** and last month?
**Assistant:** Your payslip for 2026-03 is ready. You can download it from the Payslips screen.
- intent `payslip_status` (conf. 0.1483) · action `answered` · authorization `ok` · follow-up: True
- slots now: {'month': '2026-03'}
- mock service call: GET /api/v1/payslips/me/2026-03 (X-Demo-User: EMP-1001)

**[S-1] User (EMP-1001):** what about 2026-02?
**Assistant:** Your payslip for 2026-02 is ready. You can download it from the Payslips screen.
- intent `payslip_status` (conf. 0.1459) · action `answered` · authorization `ok` · follow-up: True
- slots now: {'month': '2026-02'}
- mock service call: GET /api/v1/payslips/me/2026-02 (X-Demo-User: EMP-1001)

**[S-1] User (EMP-1001):** and my attendance?
**Assistant:** In 2026-02 you were present 19 of 20 days with 1 late arrivals.
- intent `attendance_summary` (conf. 0.7442) · action `answered` · authorization `ok` · follow-up: True
- slots now: {'month': '2026-02'}
- mock service call: GET /api/v1/attendance/summary?employee=me&month=2026-02 (X-Demo-User: EMP-1001)

**[S-1] User (EMP-1001):** thanks
**Assistant:** You are welcome!
- intent `thanks` (conf. 0.9956) · action `static` · authorization `ok` · follow-up: False
- slots now: none
- mock service call: **none**

## Transcript 2: salary refusal cannot be undone through memory (employee EMP-1001)

Own payslip (allowed) -> request about another person (refused, context wiped) -> ambiguous follow-ups (no data call) -> explicit own request works again.

**[S-1 new] User (EMP-1001):** show my payslip for last month
**Assistant:** Your payslip for 2026-03 is ready. You can download it from the Payslips screen.
- intent `payslip_status` (conf. 0.9689) · action `answered` · authorization `ok` · follow-up: False
- slots now: {'month': '2026-03'}
- mock service call: GET /api/v1/payslips/me/2026-03 (X-Demo-User: EMP-1001)

**[S-1] User (EMP-1001):** What about Rahul's salary?
**Assistant:** Sorry, I cannot share another person's pay. You can ask about your own payslip.
- intent `salary_of_other` (conf. 0.3027) · action `refused` · authorization `refuse_always` · follow-up: False
- slots now: none
- mock service call: **none**

**[S-1] User (EMP-1001):** And last month?
**Assistant:** I'm not sure I understood your question. Please rephrase it, or contact HR for assistance.
- intent `None` (conf. 0.1483) · action `fallback` · authorization `n/a` · follow-up: False
- slots now: none
- mock service call: **none**

**[S-1] User (EMP-1001):** and his?
**Assistant:** I'm not sure I understood your question. Please rephrase it, or contact HR for assistance.
- intent `None` (conf. 0.0583) · action `fallback` · authorization `n/a` · follow-up: False
- slots now: none
- mock service call: **none**

**[S-1] User (EMP-1001):** show my payslip for last month
**Assistant:** Your payslip for 2026-03 is ready. You can download it from the Payslips screen.
- intent `payslip_status` (conf. 0.9689) · action `answered` · authorization `ok` · follow-up: False
- slots now: {'month': '2026-03'}
- mock service call: GET /api/v1/payslips/me/2026-03 (X-Demo-User: EMP-1001)
