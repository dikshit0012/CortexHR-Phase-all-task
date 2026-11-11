"""Training, development-split evaluation and confidence-threshold selection. Uses the TRAINING file only.

The official test file is never read here. Model choices (word vs word+char features, C) and the threshold are made on
out-of-fold predictions of a GROUPED cross-validation: messages that share the same base phrase (see data.phrase_group) are
always in the same fold, so every dev message is a phrasing the model has not seen - the same situation as the test set.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from .data import load_training, phrase_groups, sha256_file
from .intent_model import BASELINE, DEFAULT_THRESHOLD, ModelConfig, build_pipeline, save_artifact
from .text_cleaning import clean_text

N_FOLDS = 5
TARGET_ACCEPTED_ACCURACY = 0.95      # on dev: of the messages the bot would ANSWER, at least this share must be the right intent
MIN_ACCEPTED = 50                    # ... and at least this many dev messages must be accepted (avoid a threshold chosen on noise)
THRESHOLD_GRID = np.round(np.arange(0.30, 0.951, 0.01), 2)


def out_of_fold(texts: np.ndarray, labels: np.ndarray, groups: np.ndarray, config: ModelConfig) -> Tuple[np.ndarray, List[str]]:
    classes = sorted(set(labels))
    oof = np.zeros((len(labels), len(classes)))
    for trn, dev in GroupKFold(n_splits=N_FOLDS).split(texts, labels, groups):
        model = build_pipeline(config).fit(texts[trn], labels[trn])
        if list(model.classes_) != classes:
            raise ValueError("a training fold is missing an intent; use fewer folds or more data")
        oof[dev] = model.predict_proba(texts[dev])
    return oof, classes


def _swap_one_pair(text: str, rng: random.Random) -> str:
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(w) >= 5]
    if not idx:
        return text
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def typo_robustness(texts: np.ndarray, labels: np.ndarray, groups: np.ndarray, config: ModelConfig, seed: int = 7) -> float:
    """SYNTHETIC check: one random adjacent-letter swap is injected into every dev message (after cleaning, so the typo
    dictionary cannot hide it). Returns dev accuracy. It measures robustness to UNSEEN typos, nothing else."""
    rng = random.Random(seed)
    correct = 0
    for trn, dev in GroupKFold(n_splits=N_FOLDS).split(texts, labels, groups):
        model = build_pipeline(config).fit(texts[trn], labels[trn])
        noisy = [_swap_one_pair(clean_text(t), rng) for t in texts[dev]]
        correct += int((model.predict(noisy) == labels[dev]).sum())
    return correct / len(labels)


def threshold_curve(oof: np.ndarray, classes: Sequence[str], labels: np.ndarray) -> pd.DataFrame:
    pred = np.array(classes)[oof.argmax(1)]
    conf = oof.max(1)
    ok = pred == labels
    rows = []
    for t in THRESHOLD_GRID:
        acc_mask = conf >= t
        n = int(acc_mask.sum())
        rows.append({"threshold": float(t), "answered": n, "coverage": n / len(labels),
                     "accuracy_when_answered": float(ok[acc_mask].mean()) if n else float("nan"),
                     "wrong_answers": int((~ok[acc_mask]).sum()), "fallback_rate": 1 - n / len(labels)})
    return pd.DataFrame(rows)


def select_threshold(curve: pd.DataFrame, n_messages: int) -> Tuple[float, Dict[str, Any]]:
    """Smallest threshold whose dev accuracy-when-answered reaches the target with enough answered messages."""
    good = curve[(curve["accuracy_when_answered"] >= TARGET_ACCEPTED_ACCURACY) & (curve["answered"] >= MIN_ACCEPTED)]
    if good.empty:
        return DEFAULT_THRESHOLD, {"method": "default", "reason": f"no threshold on the dev split reached {TARGET_ACCEPTED_ACCURACY:.0%} "
                                   f"accuracy with >= {MIN_ACCEPTED} answered messages; the brief's initial value was kept"}
    row = good.iloc[0]
    return float(row["threshold"]), {
        "method": "grouped-cross-validation dev split of the TRAINING data (official test set not used)",
        "rule": f"smallest threshold with dev accuracy-when-answered >= {TARGET_ACCEPTED_ACCURACY:.0%} and >= {MIN_ACCEPTED} answered dev messages",
        "dev_answered": int(row["answered"]), "dev_coverage": round(float(row["coverage"]), 4),
        "dev_accuracy_when_answered": round(float(row["accuracy_when_answered"]), 4),
        "dev_wrong_answers": int(row["wrong_answers"]), "folds": N_FOLDS, "dev_messages": int(n_messages),
    }


def expected_calibration_error(oof: np.ndarray, classes: Sequence[str], labels: np.ndarray, bins: int = 10) -> Tuple[float, pd.DataFrame]:
    pred = np.array(classes)[oof.argmax(1)]
    conf, ok = oof.max(1), (pred == labels)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(conf[m].mean() - ok[m].mean())
            ece += m.mean() * gap
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_confidence": float(conf[m].mean()), "accuracy": float(ok[m].mean())})
    return float(ece), pd.DataFrame(rows)


def compare_configs(texts, labels, groups) -> pd.DataFrame:
    grid = [("word-only, C=1 (brief baseline)", BASELINE)] + [
        (f"word-only, C={c:g}", ModelConfig(char_ngrams=None, C=c)) for c in (10.0, 30.0)] + [
        (f"word+char, C={c:g}", ModelConfig(C=c)) for c in (1.0, 10.0, 30.0)]
    rows = []
    for name, cfg in grid:
        oof, classes = out_of_fold(texts, labels, groups, cfg)
        pred = np.array(classes)[oof.argmax(1)]
        conf = oof.max(1)
        a60 = conf >= 0.60
        rows.append({"config": name, "dev_accuracy": accuracy_score(labels, pred),
                     "dev_macro_f1": f1_score(labels, pred, average="macro"),
                     "dev_accuracy_with_injected_typos": typo_robustness(texts, labels, groups, cfg),
                     "mean_max_probability": float(conf.mean()),
                     "share_answered_at_0.60": float(a60.mean()),
                     "accuracy_when_answered_at_0.60": float((pred[a60] == labels[a60]).mean()) if a60.any() else float("nan")})
    return pd.DataFrame(rows)


def train_and_save(train_csv: Path, model_path: Path, report_dir: Optional[Path] = None, config: ModelConfig = ModelConfig(),
                   run_comparison: bool = True) -> Dict[str, Any]:
    train = load_training(train_csv)
    texts = train["text"].to_numpy()
    labels = train["intent"].to_numpy()
    groups = np.array(phrase_groups(texts))

    oof, classes = out_of_fold(texts, labels, groups, config)
    curve = threshold_curve(oof, classes, labels)
    threshold, selection = select_threshold(curve, len(labels))
    ece, reliability = expected_calibration_error(oof, classes, labels)
    pred = np.array(classes)[oof.argmax(1)]
    dev = {"dev_accuracy": float(accuracy_score(labels, pred)), "dev_macro_f1": float(f1_score(labels, pred, average="macro")),
           "expected_calibration_error": round(ece, 4), "dev_groups": int(len(set(groups)))}
    selection = {**selection, **dev}

    final = build_pipeline(config).fit(texts, labels)             # the saved model sees ALL training rows, never the test set
    training = {"rows": int(len(train)), "intents": int(train["intent"].nunique()), "data_file": Path(train_csv).name,
                "data_sha256": sha256_file(train_csv)}
    meta = save_artifact(model_path, final, threshold, selection, training, config,
                         datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    out = {"metadata": meta, "threshold_curve": curve, "reliability": reliability}
    if run_comparison:
        out["config_comparison"] = compare_configs(texts, labels, groups)
    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        curve.to_csv(report_dir / "threshold_selection_curve.csv", index=False, float_format="%.4f")
        reliability.to_csv(report_dir / "dev_reliability_table.csv", index=False, float_format="%.4f")
        wrong = pred != labels
        dev_pairs = (pd.DataFrame({"true_intent": labels[wrong], "predicted_intent": pred[wrong]})
                     .value_counts().rename("dev_count").reset_index().sort_values(["dev_count", "true_intent"], ascending=[False, True]))
        dev_pairs.to_csv(report_dir / "dev_confused_pairs.csv", index=False)
        per = pd.DataFrame({"intent": labels, "ok": ~wrong, "phrase_group": groups}).groupby("intent").agg(
            dev_accuracy=("ok", "mean"), messages=("ok", "size"), distinct_base_phrasings=("phrase_group", "nunique")).reset_index()
        per.sort_values("dev_accuracy").to_csv(report_dir / "dev_per_intent_accuracy.csv", index=False, float_format="%.4f")
        if "config_comparison" in out:
            out["config_comparison"].to_csv(report_dir / "model_config_comparison_dev.csv", index=False, float_format="%.4f")
    return out
