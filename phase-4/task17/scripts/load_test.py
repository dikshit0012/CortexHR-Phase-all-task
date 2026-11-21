#!/usr/bin/env python3
"""Controlled concurrency test with an asynchronous HTTP client.

    python scripts/load_test.py --base-url http://127.0.0.1:8000 --api-key KEY --levels 1,10,25,50,100 --requests-per-user 5

Each virtual user sends --requests-per-user requests in a loop (mixed endpoints). Reports average / P95 / P99 latency,
throughput, error rate and timeouts per level. 429 (rate limited) is counted separately from errors because it is the API
protecting itself. Do NOT point this at a free hosting plan with high levels: it is a load test, keep levels small there.
"""
import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import samples as S  # noqa: E402

MIX = [("/v1/attrition/predict", {"records": [S.ATTRITION]}), ("/v1/chat", S.CHAT), ("/v1/payband/analyze", S.PAYBAND),
       ("/v1/forecast/headcount", S.FORECAST), ("/v1/resume/rank", S.RESUME)]


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else float("nan")


async def user(client, base, n, out, uid):
    for i in range(n):
        path, body = MIX[(uid + i) % len(MIX)]
        t0 = time.perf_counter()
        try:
            r = await client.post(base + path, json=body)
            out.append((r.status_code, (time.perf_counter() - t0) * 1000, False))
        except httpx.TimeoutException:
            out.append((0, (time.perf_counter() - t0) * 1000, True))
        except httpx.HTTPError:
            out.append((-1, (time.perf_counter() - t0) * 1000, False))


async def run_level(base, key, users, per_user, timeout):
    out = []
    limits = httpx.Limits(max_connections=users, max_keepalive_connections=users)
    async with httpx.AsyncClient(timeout=timeout, headers={"X-API-Key": key} if key else {}, limits=limits) as c:
        t0 = time.perf_counter()
        await asyncio.gather(*(user(c, base, per_user, out, u) for u in range(users)))
        wall = time.perf_counter() - t0
    lat = [m for s, m, _ in out if s == 200]
    ok, rl = sum(s == 200 for s, _, _ in out), sum(s == 429 for s, _, _ in out)
    to, err = sum(t for *_, t in out), sum(s not in (200, 429, 0) for s, _, _ in out)
    return {"users": users, "requests": len(out), "ok": ok, "rate_limited": rl, "errors": err, "timeouts": to,
            "avg_ms": statistics.fmean(lat) if lat else float("nan"), "p95_ms": pct(lat, 95), "p99_ms": pct(lat, 99),
            "rps": len(out) / wall, "error_rate_pct": 100 * (err + to) / len(out)}


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("CORTEXHR_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("CORTEXHR_API_KEY"))
    ap.add_argument("--levels", default="1,10,25,50")
    ap.add_argument("--requests-per-user", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args()
    if not a.base_url:
        ap.error("--base-url is required")
    base = a.base_url.rstrip("/")
    print(f"Load test against {base} (mixed endpoints, {a.requests_per_user} requests per user)\n")
    print(f"{'users':>5} {'reqs':>5} {'ok':>5} {'429':>5} {'err':>4} {'tmo':>4} {'avg ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'req/s':>7} {'err %':>6}")
    for lv in [int(x) for x in a.levels.split(",")]:
        r = await run_level(base, a.api_key, lv, a.requests_per_user, a.timeout)
        print(f"{r['users']:>5} {r['requests']:>5} {r['ok']:>5} {r['rate_limited']:>5} {r['errors']:>4} {r['timeouts']:>4} "
              f"{r['avg_ms']:>8.0f} {r['p95_ms']:>8.0f} {r['p99_ms']:>8.0f} {r['rps']:>7.1f} {r['error_rate_pct']:>6.1f}")
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
