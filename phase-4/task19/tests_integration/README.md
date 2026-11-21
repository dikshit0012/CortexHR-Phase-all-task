# Live integration tests (console ↔ the REAL Task 17 API, no mocked HTTP)

These tests start the real API (real saved model artifacts) and drive the client and all pages against it.

```bash
# Option A: start the API for you (needs the repo + a Python env with Task 17's requirements.txt installed)
export CORTEXHR_API_REPO=/path/to/CortexHR/phase-4/task17
export CORTEXHR_API_PYTHON=/path/to/venv/bin/python
python -m pytest tests_integration -q

# Option B: test an already running / deployed API
export CORTEXHR_TEST_API_URL=https://your-api  CORTEXHR_TEST_API_KEY=...
python -m pytest tests_integration -q        # server-config tests (429 / missing key) are skipped in this mode
```
Without either, the tests are skipped (never silently passed).
