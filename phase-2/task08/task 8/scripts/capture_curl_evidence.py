"""Start the API, run the README curl examples against it, and write the REAL responses to reports/task8_curl_evidence.md.

    set CORTEXHR_WORKBOOK=path\to\02_CortexHR_AI_ML_Dataset.xlsx      (PowerShell: $env:CORTEXHR_WORKBOOK="...")
    python scripts/build_artifacts.py
    python scripts/capture_curl_evidence.py

Requires `curl` (curl.exe on Windows 10/11). Nothing is faked: every response is whatever the running service returned.
"""
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
CURL = shutil.which("curl.exe") or shutil.which("curl")
JSON_H = "Content-Type: application/json"
STATUS = "\nHTTP %{http_code}"


def curl(*args: str) -> str:
    out = subprocess.run([CURL, "-s", *args], capture_output=True, text=True, cwd=ROOT, timeout=120).stdout
    body, _, status = out.rpartition("\nHTTP ")
    try:
        body = json.dumps(json.loads(body), indent=2)
    except ValueError:
        pass
    return f"{body}\n(HTTP {status.strip()})"


def main() -> int:
    if not CURL:
        print("curl not found", file=sys.stderr)
        return 1
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)], cwd=ROOT,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{BASE}/v1/health", timeout=1).read()
                break
            except Exception:
                if server.poll() is not None:
                    print("server exited during startup:\n" + server.stdout.read(), file=sys.stderr)
                    return 1
                time.sleep(0.5)
        else:
            print("server did not become ready", file=sys.stderr)
            return 1
        cases = [
            ("health", ["-w", STATUS, f"{BASE}/v1/health"]),
            ("models", ["-w", STATUS, f"{BASE}/v1/models"]),
            ("attendance", ["-w", STATUS, "-X", "POST", f"{BASE}/v1/anomaly/attendance", "-H", JSON_H, "-d", "@examples/attendance_request.json"]),
            ("payroll", ["-w", STATUS, "-X", "POST", f"{BASE}/v1/anomaly/payroll", "-H", JSON_H, "-d", "@examples/payroll_request.json"]),
            ("attendance batch", ["-w", STATUS, "-X", "POST", f"{BASE}/v1/anomaly/attendance/batch", "-H", JSON_H, "-d", "@examples/attendance_batch_request.json"]),
            ("payroll batch", ["-w", STATUS, "-X", "POST", f"{BASE}/v1/anomaly/payroll/batch", "-H", JSON_H, "-d", "@examples/payroll_batch_request.json"]),
            ("chat - approved FAQ answer", ["-w", STATUS, "-X", "POST", f"{BASE}/v1/chat", "-H", JSON_H, "-d", "@examples/chat_request.json"]),
            ("chat - safe fallback", ["-w", STATUS, "-X", "POST", f"{BASE}/v1/chat", "-H", JSON_H, "-d", "@examples/chat_fallback_request.json"]),
            ("chat - missing message (expect 422)", ["-w", STATUS, "-X", "POST", f"{BASE}/v1/chat", "-H", JSON_H, "-d", "{}"]),
            ("invalid attendance request (expect 422)", ["-w", STATUS, "-X", "POST", f"{BASE}/v1/anomaly/attendance", "-H", JSON_H, "-d", "@examples/invalid_attendance_request.json"]),
            ("empty batch (expect 422)", ["-w", STATUS, "-X", "POST", f"{BASE}/v1/anomaly/payroll/batch", "-H", JSON_H, "-d", '{"records":[]}']),
            ("unknown detector (expect 404)", ["-w", STATUS, "-X", "POST", f"{BASE}/v1/anomaly/turnover/batch", "-H", JSON_H, "-d", '{"records":[]}']),
        ]
        lines = ["# Task 8 - curl evidence (captured from the running service)", "",
                 f"Server: `python -m uvicorn app.main:app --port {PORT}` on {platform.system()}, Python {platform.python_version()}. "
                 "Every response below is real output of that run.", ""]
        for title, args in cases:
            shown = " ".join(f"'{a}'" if (" " in a or a.startswith("{")) else a
                             for a in ["curl", "-s", *[x for x in args if x not in ("-w", STATUS)]])
            lines += [f"## {title}", "```", f"$ {shown}", curl(*args), "```", ""]
        (ROOT / "reports" / "task8_curl_evidence.md").write_text("\n".join(lines), encoding="utf-8")
        print("wrote reports/task8_curl_evidence.md")
    finally:
        server.terminate()
        try:
            log = server.communicate(timeout=10)[0]
        except Exception:
            server.kill()
            log = ""
        print("---- uvicorn log ----\n" + log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
