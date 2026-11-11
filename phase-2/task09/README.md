# Task 9 - HR FAQ intent classifier

Two Task 9 copies are present. **Use [`cortexhr_task9/task 9/README.md`](cortexhr_task9/task%209/README.md) as the canonical runnable project.** `cortexhr_task9_audit/task 9` is a retained audit copy and is not the path integrated with Task 8.

The canonical project includes its source, tests, and starter chatbot CSVs. Its trained joblib model is intentionally not included: train it locally in the same Python environment used to serve Task 8. Task 9 adds no runtime dependencies beyond Task 8's requirements.

From the repository root in Windows PowerShell:

```powershell
python -m pip install -r 'phase-2\task08\task 8\requirements.txt'
python 'phase-2\task09\cortexhr_task9\task 9\scripts\train_intent_model.py'
python 'phase-2\task09\cortexhr_task9\task 9\scripts\evaluate_intent_model.py'
python -m pytest 'phase-2\task09\cortexhr_task9\task 9\tests'
```

To serve the chat route, follow [`task08/task 8/README.md`](../task08/task%208/README.md) and set `CORTEXHR_TASK9_ROOT` to the canonical `task 9` directory if automatic discovery does not locate it. The model artifact must be built with compatible scikit-learn, NumPy, SciPy, and joblib versions in the serving environment. Joblib artifacts are executable pickle data; load only artifacts built from this source and trusted inputs.

Task 9 is a narrow intent/FAQ classifier, not an LLM or an authenticated HR product. Review the nested README's limitations before using it.
