"""Builds and executes notebooks/task16_headcount_analysis.ipynb from the pipeline (a thin walkthrough; logic lives in src/)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
cells = [
    nbf.v4.new_markdown_cell("# Task 16 - headcount forecast with uncertainty\nWalkthrough of the pipeline in `src/cortexhr_task16`. Re-run with `python scripts/build_notebook.py`."),
    nbf.v4.new_code_cell("import sys, warnings; sys.path.insert(0, '../src'); warnings.filterwarnings('ignore')\nimport pandas as pd\nfrom IPython.display import Image, display\nfrom cortexhr_task16 import forecasting, exploratory_analysis as ea, config\nres = forecasting.run(write=False)\nprint('selected model:', res['model'], '| trained through', res['trained_through'])"),
    nbf.v4.new_markdown_cell("## 1. Data quality and EDA"),
    nbf.v4.new_code_cell("print(res['quality'])\neda = ea.analyse(res['hist'], res['events'])\ndisplay(eda['annual'].round(1)); display(eda['by_month'].round(2)); display(eda['unusual'])"),
    nbf.v4.new_code_cell("display(Image(str(config.CHARTS / 'historical_headcount.png')))"),
    nbf.v4.new_markdown_cell("## 2. Model selection (cross-validation inside training data only)"),
    nbf.v4.new_code_cell("display(res['cv_final'].round(3))"),
    nbf.v4.new_markdown_cell("## 3. Six-month hold-out backtest"),
    nbf.v4.new_code_cell("results, metrics = forecasting.backtest_tables(res)\ndisplay(metrics[metrics.department_id=='ALL'].round(2)); display(results[results.department_id=='ALL'].round(1))"),
    nbf.v4.new_code_cell("display(Image(str(config.CHARTS / 'backtest_comparison.png'))); display(Image(str(config.CHARTS / 'model_performance.png')))"),
    nbf.v4.new_markdown_cell("## 4. Forecast Oct 2026 - Mar 2027 and reliability"),
    nbf.v4.new_code_cell("display(res['headcount_fc']['ALL'].round(1)); display(res['exits_fc'].round(2)); display(res['reliability']['table'].round(2))"),
    nbf.v4.new_code_cell("display(Image(str(config.CHARTS / 'six_month_forecast.png')))"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebooks" / "task16_headcount_analysis.ipynb"
NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(out.parent)}}).execute()
nbf.write(nb, out)
print("wrote", out)
