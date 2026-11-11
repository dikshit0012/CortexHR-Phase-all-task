#!/usr/bin/env python
"""Re-run the whole Task 1 audit headlessly.

    python scripts/run_audit.py

Executes notebooks/01_data_audit.ipynb top to bottom and saves the executed copy in place.
Outputs: reports/figures/*.png, reports/tables/*.csv, reports/findings.json,
         reports/ydata_profile_attrition.html, docs/data_dictionary.md, docs/decisions.md
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "01_data_audit.ipynb"
DATA = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"

if not DATA.exists():
    sys.exit(f"Missing data file: {DATA}\nCopy the master workbook into data/raw/ and try again.")

nb = nbformat.read(NB, as_version=4)
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
nbformat.write(nb, NB)
print("Audit complete. See reports/ and docs/.")
