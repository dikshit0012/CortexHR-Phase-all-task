"""Regenerate contract_snapshot.json from a RUNNING CortexHR API (its /openapi.json plus the documented request examples).
The console's forms are built from this snapshot; tests_integration compares it with the live API so drift is caught.

Usage:  python scripts/sync_contract.py --base-url https://your-api --api-key KEY [--out contract_snapshot.json]
"""
import argparse, json, re, sys
from pathlib import Path
import requests


def field_info(prop: dict) -> dict:
    opts = prop.get("anyOf") or [prop]
    core = next((o for o in opts if o.get("type") != "null"), opts[0])
    info = {"type": core.get("type", "string"), "format": core.get("format")}
    for k_src, k_dst in (("minimum", "min"), ("maximum", "max"), ("maxLength", "max_length"), ("enum", "enum")):
        if k_src in core:
            info[k_dst] = core[k_src]
    return {k: v for k, v in info.items() if v is not None}


def schema_fields(spec: dict, name: str) -> dict:
    s = spec["components"]["schemas"][name]
    req = set(s.get("required", []))
    return {k: {**field_info(v), "required": k in req} for k, v in s["properties"].items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base-url", required=True); ap.add_argument("--api-key", default="")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "contract_snapshot.json")); a = ap.parse_args()
    spec = requests.get(a.base_url.rstrip("/") + "/openapi.json", timeout=30).json()
    ex = lambda p: spec["paths"][p]["post"]["requestBody"]["content"]["application/json"].get("example")
    H = {"X-API-Key": a.api_key}
    bad = requests.post(a.base_url.rstrip("/") + "/v1/forecast/headcount", json={"department_id": "ZZZ"}, headers=H, timeout=30).json()
    depts = re.findall(r"\b(?:ALL|D\d+)\b", bad.get("error", {}).get("message", "").split("Available:")[-1])
    snap = {
        "api_version": spec["info"]["version"],
        "paths": sorted(spec["paths"]),
        "attrition": {"fields": schema_fields(spec, "AttritionRecord"), "sample": ex("/v1/attrition/predict")["records"][0]},
        "attendance": {"fields": schema_fields(spec, "AttendanceRecord")},
        "payroll": {"fields": schema_fields(spec, "PayrollRecord")},
        "payband": {"fields": schema_fields(spec, "PayBandRequest"), "sample": ex("/v1/payband/analyze")},
        "resume_job": {"fields": schema_fields(spec, "JobIn")},
        "forecast_departments": depts,
    }
    snap["attrition"]["sample"].pop("record_ref", None)
    Path(a.out).write_text(json.dumps(snap, indent=1, sort_keys=True))
    print("wrote", a.out, "| paths:", len(snap["paths"]), "| attrition fields:", len(snap["attrition"]["fields"]), "| departments:", depts)


if __name__ == "__main__":
    sys.exit(main())
