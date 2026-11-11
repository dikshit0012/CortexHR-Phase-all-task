#!/usr/bin/env python
"""Day 2 / Task 2: build the feature table, fit the pipeline on TRAIN only, transform TRAIN and TEST.

    python scripts/run_pipeline.py              # confirmed-safe features only (default)
    python scripts/run_pipeline.py --include-unconfirmed   # also add dept_attrition_rate_prior_12m

Writes data/processed/feature_pipeline.joblib (fitted pipeline), reports/tables/day2_pipeline_run.json
and prints a short report. Nothing here is a model: training a model is Task 3.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cortexhr.audit import TARGET, load_attrition                      # noqa: E402
from cortexhr.features import (FORBIDDEN_COLUMNS, build_pipeline, feature_columns,   # noqa: E402
                               output_feature_names, prepare_training_rows, select_features)
from cortexhr.monthly import add_monthly_features, load_monthly_tables  # noqa: E402
from cortexhr.paths import DATA_PROCESSED, TABLES, WORKBOOK            # noqa: E402


def main(include_unconfirmed: bool = False, seed: int = 42) -> dict:
    if not WORKBOOK.exists():
        sys.exit(f"Missing data file: {WORKBOOK}\nCopy the master workbook into data/raw/ and try again.")

    raw = load_attrition()
    attendance, payroll = load_monthly_tables()

    # 1. row-level Task 1 decisions (duplicates, negative tenure on hold)
    rows, row_report = prepare_training_rows(raw)
    rows = rows.reset_index(drop=True)

    # 2. point-in-time monthly features (3 completed months before each observation date)
    table, join_diag = add_monthly_features(rows, attendance, payroll)
    assert len(table) == len(rows), "monthly join changed the number of employee rows"

    # 3. inputs X and target y are separate objects from here on
    X = select_features(table, include_unconfirmed=include_unconfirmed)
    y = table[TARGET]
    assert not (set(X.columns) & FORBIDDEN_COLUMNS), "forbidden column reached X"

    # 4. split FIRST, then fit on train only
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    pipe = build_pipeline(include_unconfirmed=include_unconfirmed)
    pipe.fit(X_train)                                  # learns medians, scaling, categories from TRAIN
    Xtr, Xte = pipe.transform(X_train), pipe.transform(X_test)   # TEST only transformed, never fitted

    for name, arr in (("train", Xtr), ("test", Xte)):
        assert np.issubdtype(arr.dtype, np.number), f"{name} output is not numeric"
        assert np.isfinite(arr).all(), f"{name} output contains NaN or infinite values"

    names = output_feature_names(pipe)
    summary = {
        "include_unconfirmed": include_unconfirmed,
        "row_report": row_report,
        "monthly_join": join_diag,
        "input_columns": {k: v for k, v in feature_columns(include_unconfirmed).items()},
        "n_input_columns": X.shape[1],
        "train_shape": list(Xtr.shape), "test_shape": list(Xte.shape),
        "n_output_columns": len(names),
        "train_leaver_rate": round(float(y_train.mean()), 4), "test_leaver_rate": round(float(y_test.mean()), 4),
        "missing_indicator_columns": [n for n in names if "missingindicator" in n],
        "flag_columns": [n for n in names if n.startswith("flags__")],
        "nan_in_output": int(np.isnan(Xtr).sum() + np.isnan(Xte).sum()),
        "inf_in_output": int(np.isinf(Xtr).sum() + np.isinf(Xte).sum()),
        "seed": seed,
    }

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, DATA_PROCESSED / "feature_pipeline.joblib")
    (TABLES / "day2_pipeline_run.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"rows after Task 1 row rules : {row_report['rows_in']} -> {row_report['rows_out']} "
          f"({row_report['duplicates_dropped']} duplicates, {row_report['negative_tenure_excluded']} negative-tenure rows left out)")
    print(f"monthly join                : {join_diag['rows_in']} rows in, {join_diag['rows_out']} out; "
          f"{join_diag['rows_with_attendance_data']} rows have monthly data, {join_diag['rows_without_attendance_data']} do not")
    print(f"payroll duplicate rows removed: {join_diag['duplicate_rows_removed']['payroll_monthly_features']}")
    print(f"input columns               : {X.shape[1]}   (target and forbidden columns excluded)")
    print(f"train / test                : {Xtr.shape} / {Xte.shape}   leaver rate {summary['train_leaver_rate']} / {summary['test_leaver_rate']}")
    print(f"output columns              : {len(names)}   NaN: {summary['nan_in_output']}   inf: {summary['inf_in_output']}")
    print(f"saved fitted pipeline       : {DATA_PROCESSED / 'feature_pipeline.joblib'}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--include-unconfirmed", action="store_true",
                    help="also use dept_attrition_rate_prior_12m (not confirmed safe, open item O2)")
    main(ap.parse_args().include_unconfirmed)
