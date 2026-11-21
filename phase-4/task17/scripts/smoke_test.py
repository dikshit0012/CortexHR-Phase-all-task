#!/usr/bin/env python3
"""Smoke test against a RUNNING CortexHR API (local or deployed).

    python scripts/smoke_test.py --base-url https://YOUR-DEPLOYED-API --api-key "$CORTEXHR_API_KEY"

Every check validates HTTP status, JSON, required fields and basic semantic sanity (probabilities in [0,1], ordered intervals,
explanations present). An endpoint that is unavailable (503) is reported as FAIL, or as SKIP with --allow-unavailable (never PASS).
Exit code 0 only when nothing failed.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import samples as S  # noqa: E402  (synthetic sample inputs shared with the test-suite)


def need(cond, msg):
    if not cond:
        raise AssertionError(msg)


def c_health(j):
    need(j["application"] == "up" and j["api_version"], "application/api_version missing")
    need(set(j["models_available"]).isdisjoint(j["models_unavailable"]), "availability lists overlap")
    return f"{len(j['models_available'])}/7 models loaded"


def c_ready(j):
    need("ready" in j and "blocking" in j, "ready/blocking missing")
    need(j["ready"] is True, f"not ready, blocking={j['blocking']}")
    return "ready" + (f" (degraded: {','.join(j['degraded'])})" if j["degraded"] else "")


def c_attr(j):
    r = j["results"][0]
    need(0 <= r["attrition_probability_12m"] <= 1 and j["model_version"], "bad probability / version")
    return f"p={r['attrition_probability_12m']:.3f} ({j['model_version']})"


def c_anom(j):
    r = j["results"][0]
    need(0 <= r["anomaly_score"] <= 1 and r["flagged"] in ("Y", "N") and r["model_version"], "bad anomaly result")
    return f"score={r['anomaly_score']} flagged={r['flagged']} reason={r.get('reason_code')}"


def c_chat(j):
    need(j["reply"] and "fallback" in j and j["model_version"], "reply/fallback missing")
    need(not j["fallback"] and j["intent"] == "notice_period_policy", f"expected FAQ answer, got intent={j['intent']} fallback={j['fallback']}")
    return f"intent={j['intent']} conf={j['confidence']}"


def c_resume(j):
    rk = j["ranking"]
    need(len(rk) == 2 and rk[0]["resume_id"] == "R-1" and rk[0]["final_score"] > rk[1]["final_score"], "relevant resume not ranked first")
    need(rk[0]["matched_skills"] and "fairness_note" in j, "explanation/fairness note missing")
    return f"top={rk[0]['resume_id']} score={rk[0]['final_score']:.3f} matched={','.join(rk[0]['matched_skills'])}"


def c_payband(j):
    need(j["predicted_salary_mid"] > 0 and j["selected_model"] and j["band_status"], "bad pay-band result")
    lo, hi = j["recommended_salary_low"], j["recommended_salary_high"]
    need(lo is None or lo <= j["predicted_salary_mid"] <= hi, "band does not contain the midpoint")
    return f"mid={j['predicted_salary_mid']} band={'n/a' if lo is None else f'{lo}-{hi}'}"


def c_forecast(j):
    f = j["forecast"]
    need(len(f) == 6 and j["mode"] in ("live", "precomputed") and j["trained_through"], "forecast shape/mode wrong")
    need(all(p["lower_80"] <= p["expected_headcount"] <= p["upper_80"] for p in f), "interval not ordered")
    return f"{j['mode']} {f[0]['month']}..{f[-1]['month']} first={f[0]['expected_headcount']}"


CHECKS = [
    ("GET  /v1/health", "GET", "/v1/health", None, c_health, False),
    ("GET  /v1/ready", "GET", "/v1/ready", None, c_ready, False),
    ("POST /v1/attrition/predict", "POST", "/v1/attrition/predict", {"records": [S.ATTRITION]}, c_attr, True),
    ("POST /v1/anomaly/attendance", "POST", "/v1/anomaly/attendance", {"records": [S.ATTENDANCE]}, c_anom, True),
    ("POST /v1/anomaly/payroll", "POST", "/v1/anomaly/payroll", {"records": [S.PAYROLL]}, c_anom, True),
    ("POST /v1/attendance/anomaly", "POST", "/v1/attendance/anomaly", {"records": [S.ATTENDANCE]}, c_anom, True),
    ("POST /v1/payroll/anomaly", "POST", "/v1/payroll/anomaly", {"records": [S.PAYROLL]}, c_anom, True),
    ("POST /v1/chat", "POST", "/v1/chat", S.CHAT, c_chat, True),
    ("POST /v1/resume/rank", "POST", "/v1/resume/rank", S.RESUME, c_resume, True),
    ("POST /v1/payband/analyze", "POST", "/v1/payband/analyze", S.PAYBAND, c_payband, True),
    ("POST /v1/salary/predict", "POST", "/v1/salary/predict", S.PAYBAND, c_payband, True),
    ("POST /v1/forecast/headcount", "POST", "/v1/forecast/headcount", S.FORECAST, c_forecast, True),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("CORTEXHR_BASE_URL"), help="e.g. https://cortexhr.onrender.com")
    ap.add_argument("--api-key", default=os.environ.get("CORTEXHR_API_KEY"), help="or env CORTEXHR_API_KEY")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per request (a sleeping free host may need ~60 s)")
    ap.add_argument("--allow-unavailable", action="store_true", help="report 503 model_unavailable as SKIP instead of FAIL")
    a = ap.parse_args()
    if not a.base_url:
        ap.error("--base-url (or CORTEXHR_BASE_URL) is required")
    base = a.base_url.rstrip("/")
    headers = {"X-API-Key": a.api_key} if a.api_key else {}
    rows, passed, failed, skipped = [], 0, 0, 0
    with httpx.Client(timeout=a.timeout, headers=headers, follow_redirects=False) as c:
        for name, method, path, body, check, _ in CHECKS:
            t0, status, result, verdict = time.perf_counter(), "-", "", "FAIL"
            try:
                r = c.request(method, base + path, json=body)
                status = str(r.status_code)
                try:
                    j = r.json()
                except ValueError:
                    raise AssertionError("response is not JSON")
                if r.status_code == 503 and j.get("error", {}).get("code") == "model_unavailable" and a.allow_unavailable:
                    verdict, result = "SKIP", "model unavailable on this server (not counted as pass)"
                elif r.status_code == 503 and path == "/v1/ready":
                    raise AssertionError(f"not ready: blocking={j.get('blocking')}")
                else:
                    need(r.status_code == 200, f"HTTP {r.status_code}: {j.get('error', {}).get('message', '')[:120]}")
                    result, verdict = check(j), "PASS"
            except httpx.TimeoutException:
                result = f"TIMEOUT after {a.timeout:.0f}s"
            except Exception as exc:
                result = f"{type(exc).__name__}: {str(exc)[:140]}"
            ms = (time.perf_counter() - t0) * 1000
            passed += verdict == "PASS"; failed += verdict == "FAIL"; skipped += verdict == "SKIP"
            rows.append((name, status, f"{ms:7.0f} ms", f"{verdict}: {result}"))
    w = max(len(r[0]) for r in rows)
    print(f"\nSmoke test against {base}\n")
    print(f"{'Endpoint'.ljust(w)} | Status | Response Time | Result")
    print("-" * (w + 60))
    for n, s, t, res in rows:
        print(f"{n.ljust(w)} | {s:>6} | {t:>13} | {res}")
    print(f"\nTOTAL TESTS : {len(rows)}\nPASSED      : {passed}\nFAILED      : {failed}\nSKIPPED     : {skipped}")
    overall = "PASS" if failed == 0 and skipped == 0 else ("FAIL" if failed else "PASS WITH SKIPS (not all endpoints verified)")
    print(f"OVERALL RESULT: {overall}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
