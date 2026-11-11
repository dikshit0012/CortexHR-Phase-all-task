"""Mock HR back-end for the chatbot (CX-10). Run:  uvicorn mock_server:app --port 9000
Pretend to be a logged-in user with a header:  X-Demo-User: EMP-0188   (see mock_data/demo_users.csv for ids and roles)
It answers exactly the calls listed in chatbot_responses.backend_or_service_call, so your bot never needs a real company system."""
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
D = Path(__file__).parent / "mock_data"
emp, bal, lv, pay, hol, att, usr = (pd.read_csv(D / f"{n}.csv") for n in ["employees","leave_balances","leave_requests","payslips","holidays","attendance_monthly","demo_users"])
app = FastAPI(title="Zenith Nova mock HR services")
def me(x_demo_user: str | None):
    r = usr[usr.employee_id == x_demo_user]
    if not x_demo_user or r.empty: raise HTTPException(401, "Send header X-Demo-User with a valid employee_id")
    return x_demo_user, r.role.iloc[0]
@app.get("/api/v1/holidays")
def holidays(): return hol.to_dict("records")
@app.get("/api/v1/employees/me")
def emp_me(x_demo_user: str | None = Header(None)):
    e, _ = me(x_demo_user); return emp[emp.employee_id == e].iloc[0].to_dict()
@app.get("/api/v1/employees/me/team")
def team(x_demo_user: str | None = Header(None)):
    e, _ = me(x_demo_user); return emp[emp.manager_id == e][["employee_id","full_name","role_id"]].to_dict("records")
@app.get("/api/v1/leave/balance")
def balance(employee: str = "me", x_demo_user: str | None = Header(None)):
    e, _ = me(x_demo_user); return bal[bal.employee_id == e].iloc[0].to_dict()
@app.get("/api/v1/leave/requests")
def requests(employee: str = "me", limit: int = 5, x_demo_user: str | None = Header(None)):
    e, _ = me(x_demo_user); return lv[lv.employee_id == e].sort_values("applied_on", ascending=False).head(limit).to_dict("records")
@app.get("/api/v1/payslips/me/{month}")
def payslip(month: str, x_demo_user: str | None = Header(None)):
    e, _ = me(x_demo_user); r = pay[(pay.employee_id == e) & (pay.run_id == f"PR-{month}")]
    if r.empty: raise HTTPException(404, "No payslip for that month (use YYYY-MM within the last 6 months)")
    return r.iloc[0].to_dict()
@app.get("/api/v1/attendance/summary")
def attendance(month: str, employee: str = "me", x_demo_user: str | None = Header(None)):
    e, _ = me(x_demo_user); r = att[(att.employee_id == e) & att.period_end_date.str.startswith(month)]
    if r.empty: raise HTTPException(404, "No attendance for that month (use YYYY-MM)")
    return r.iloc[0].to_dict()
