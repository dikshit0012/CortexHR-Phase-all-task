#!/usr/bin/env python
"""Run any project notebook top to bottom and save the executed copy in place.

    python scripts/run_notebook.py notebooks/02_baselines.ipynb

(scripts/run_audit.py does the same for the Task 1 notebook.)
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"

if len(sys.argv) != 2:
    sys.exit("Usage: python scripts/run_notebook.py notebooks/<name>.ipynb")
nb_path = (ROOT / sys.argv[1]).resolve()
if not nb_path.exists():
    sys.exit(f"Notebook not found: {nb_path}")
if not DATA.exists():
    sys.exit(f"Missing data file: {DATA}\nCopy the master workbook into data/raw/ and try again.")

nb = nbformat.read(nb_path, as_version=4)
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}}).execute()
nbformat.write(nb, nb_path)
print(f"Executed {nb_path.name}. See reports/ for tables and figures.")
