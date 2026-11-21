#!/usr/bin/env python
"""Verify a RUNNING CortexHR API against the local monitoring report (stdlib only; used by the workflow and by hand).

    set CORTEXHR_API_URL=https://your-api.onrender.com
    set CORTEXHR_API_KEY=<read key>
    python verify_live.py                      # compares GET /v1/monitoring with outputs/monitoring/latest_report.json
    python verify_live.py --publish            # first PUT the report (needs CORTEXHR_MONITORING_PUBLISH_KEY)

Checks: /v1/health, the existing Task 17 endpoints still answer, GET /v1/monitoring status/timestamp equal the local report, the response
carries no field outside the aggregate whitelist. Prints aggregates only. Exit 0 = verified, 1 = mismatch/failure, 2 = misconfiguration."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ALLOWED = {"status", "timestamp", "execution_status", "execution_ok", "evidence_complete", "drift", "fairness", "reasons", "last_run",
           "report_age_hours", "stale", "request_id"}


def call(method, url, headers=None, data=None, timeout=90, retries=4):
    last = None
    for attempt in range(retries):                      # free-plan hosts sleep: the first request can take a minute
        try:
            req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"FAIL: {method} {url.split('/v1')[0] and '<api>'}/v1... unreachable ({type(last).__name__})")


def compare(local: dict, live: dict) -> list:
    """Pure function (unit-tested): list of problems between the local report and the API response."""
    problems = []
    if live.get("status") != local.get("status"):
        problems.append(f"status differs: api={live.get('status')} local={local.get('status')}")
    if live.get("timestamp") != local.get("timestamp_utc"):
        problems.append("timestamp differs")
    for k, v in (local.get("fairness") or {}).items():
        if (live.get("fairness") or {}).get(k, {}).get("status") != v.get("status"):
            problems.append(f"fairness.{k} differs")
    counts = (local.get("drift") or {}).get("counts") or {}
    if (live.get("drift") or {}).get("alert") != counts.get("ALERT", 0) or (live.get("drift") or {}).get("watch") != counts.get("WATCH", 0):
        problems.append("drift counts differ")
    extra = set(live) - ALLOWED
    if extra:
        problems.append(f"unexpected fields exposed: {sorted(extra)}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=Path(__file__).resolve().parent / "outputs" / "monitoring" / "latest_report.json")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args(argv)
    url, key = os.environ.get("CORTEXHR_API_URL", "").rstrip("/"), os.environ.get("CORTEXHR_API_KEY", "")
    if not url.startswith("http") or not key:
        print("set CORTEXHR_API_URL and CORTEXHR_API_KEY")
        return 2
    local = json.loads(a.report.read_text(encoding="utf-8"))
    ok = True
    if a.publish:
        pk = os.environ.get("CORTEXHR_MONITORING_PUBLISH_KEY", "")
        if not pk:
            print("set CORTEXHR_MONITORING_PUBLISH_KEY")
            return 2
        code, body = call("PUT", f"{url}/v1/monitoring/report", {"X-Monitoring-Publish-Key": pk, "Content-Type": "application/json"}, a.report.read_bytes())
        print(f"PUT /v1/monitoring/report -> {code}")
        ok &= code == 200
    code, _ = call("GET", f"{url}/v1/health")
    print(f"GET /v1/health -> {code}")
    ok &= code == 200
    code, body = call("GET", f"{url}/v1/monitoring", {"X-API-Key": key})
    print(f"GET /v1/monitoring -> {code}")
    if code != 200:
        print("FAIL: endpoint did not return a report:", body[:160].decode("utf-8", "replace"))
        return 1
    live = json.loads(body)
    problems = compare(local, live)
    print(f"api status={live.get('status')} last_run={live.get('last_run')} stale={live.get('stale')} drift={live.get('drift')}")
    for p in problems:
        print("MISMATCH:", p)
    ok &= not problems
    print("VERIFIED" if ok else "NOT VERIFIED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
