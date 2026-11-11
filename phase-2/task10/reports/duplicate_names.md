# Task 10: how duplicate employee names are handled

Generated 2026-10-30 03:14 UTC by scripts/run_evidence.py. Reference date for 'last month' = 2026-04-15. mock_data is SYNTHETIC (the real mock_data CSVs were not supplied); pay values are sentinels and are never printed.

1. The mock HR service has **no employee directory and no endpoint that returns another person's data** (every route is scoped to `X-Demo-User`; the `employee` query parameter is ignored). So the chatbot has no authorised use for another person's identity.
2. Names are therefore recognised in order to **refuse**, never to look someone up. A request about a named person for any protected intent is declined before any service call, and known, unknown and duplicate names get the same answer (no name oracle).
3. The directory-based resolver is still implemented and tested for a future authorised flow. Actual output for the synthetic duplicate 'Rahul Sharma': status = `ambiguous`, 2 matches.
4. The clarification it would ask uses only the department: "I found more than one employee with that name. Could you clarify which one you mean (for example in Engineering, Finance)?"  (no ids, pay, contact details).
5. In chat, `show attendance for Rahul Sharma` actually returns: "I'm not sure I understood your question. Please rephrase it, or contact HR for assistance."