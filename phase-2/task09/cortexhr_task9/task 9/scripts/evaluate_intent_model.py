"""Evaluate the SAVED model on the official chatbot_test_set.csv (read-only use: nothing here changes the model or threshold).

Run from the `task 9` folder:   python scripts/evaluate_intent_model.py
Writes into reports/: intent_evaluation.json, classification_report.csv, confusion_matrix.csv, confusion_matrix.png,
confused_pairs.csv, misclassified_examples.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402

from cortexhr_task9.data import load_test, load_training, phrase_group, sha256_file  # noqa: E402
from cortexhr_task9.chat_service import ChatService  # noqa: E402
from cortexhr_task9.faq import load_responder  # noqa: E402
from cortexhr_task9.intent_model import load_artifact  # noqa: E402
from cortexhr_task9.paths import DEFAULT_FAQ, DEFAULT_MODEL, ENV_MODEL, INTENT_FAQ_MAP, REPORTS, TEST_CSV, TRAIN_CSV, env_path  # noqa: E402
from cortexhr_task9.text_cleaning import clean_text  # noqa: E402


def main() -> None:
    model = load_artifact(env_path(ENV_MODEL, DEFAULT_MODEL))
    test, train = load_test(TEST_CSV), load_training(TRAIN_CSV)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # --- contamination check: does the official test set overlap the training data?
    train_clean = set(train["text"].map(clean_text))
    train_cores = set(train["text"].map(phrase_group))
    exact = int(test["text"].map(clean_text).isin(train_clean).sum())
    core = int(test["text"].map(phrase_group).isin(train_cores).sum())
    assert model.metadata["training"]["data_sha256"] == sha256_file(TRAIN_CSV), "artifact was not trained on this training file"

    proba = model.predict_proba(test["text"].tolist())
    classes = np.array(model.classes)
    pred = classes[proba.argmax(1)]
    conf = proba.max(1)
    y = test["expected_intent"].to_numpy()
    labels = sorted(set(y) | set(classes))

    rep = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(REPORTS / "classification_report.csv", float_format="%.4f")
    cm = confusion_matrix(y, pred, labels=labels)
    pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels]).to_csv(REPORTS / "confusion_matrix.csv")

    pairs = pd.DataFrame([{"true_intent": labels[i], "predicted_intent": labels[j], "count": int(cm[i, j])}
                          for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
                         ).sort_values(["count", "true_intent", "predicted_intent"], ascending=[False, True, True])
    pairs.to_csv(REPORTS / "confused_pairs.csv", index=False)
    # unordered pairs (A->B plus B->A), because 'which two intents get confused' is symmetric in practice
    unordered = {}
    for _, r in pairs.iterrows():
        k = tuple(sorted((r.true_intent, r.predicted_intent)))
        unordered[k] = unordered.get(k, 0) + int(r["count"])
    unordered_df = pd.DataFrame([{"intent_a": a, "intent_b": b, "confusions_both_directions": n} for (a, b), n in unordered.items()]
                                ).sort_values(["confusions_both_directions", "intent_a"], ascending=[False, True])

    order = np.argsort(-proba, axis=1)
    wrong = pred != y
    mis = pd.DataFrame({"test_id": test["test_id"], "text": test["text"], "true_intent": y, "predicted_intent": pred,
                        "confidence": conf.round(4),
                        "second_choice": classes[order[:, 1]], "second_confidence": np.take_along_axis(proba, order[:, 1:2], 1)[:, 0].round(4),
                        "true_intent_probability": [round(float(proba[i, list(classes).index(t)]), 4) if t in classes else 0.0 for i, t in enumerate(y)]})[wrong]
    mis.sort_values("confidence", ascending=False).to_csv(REPORTS / "misclassified_examples.csv", index=False)

    thr = model.threshold
    answered = conf >= thr
    result = {
        "model_version": model.version, "scikit_learn_version": model.metadata["sklearn_version"],
        "test_rows": int(len(test)), "test_data_sha256": sha256_file(TEST_CSV),
        "training_rows": int(len(train)),
        "overlap_with_training": {"exact_after_cleaning": exact, "same_base_phrase_after_filler_removal": core},
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted", labels=labels, zero_division=0)), 4),
        "misclassified": int(wrong.sum()),
        "top_confused_pairs_directed": pairs.head(10).to_dict("records"),
        "top_confused_pairs_both_directions": unordered_df.head(10).to_dict("records"),
        "mean_confidence_correct": round(float(conf[~wrong].mean()), 4), "mean_confidence_wrong": round(float(conf[wrong].mean()), 4),
        "threshold_info_only": {
            "threshold": thr, "note": "Reported for information. The threshold was fixed from the training dev split BEFORE this evaluation and was not changed.",
            "answered": int(answered.sum()), "fallback": int((~answered).sum()),
            "accuracy_when_answered": round(float((pred[answered] == y[answered]).mean()), 4) if answered.any() else None,
            "wrong_answers_above_threshold": int((wrong & answered).sum()),
            "correct_predictions_sent_to_fallback": int((~wrong & ~answered).sum())},
    }

    # --- end-to-end behaviour of the chatbot (model + threshold + FAQ rules) on the same messages. Reporting only.
    chat = ChatService(model, load_responder(DEFAULT_FAQ, INTENT_FAQ_MAP, model.classes))
    replies = [chat.reply(t) for t in test["text"]]
    answered_ids = [i for i, r in enumerate(replies) if not r.fallback]
    wrong_answers = [i for i in answered_ids if replies[i].intent != y[i]]
    end_to_end = {
        "note": "Full pipeline (classifier + threshold + approved-FAQ rules). A 'wrong approved answer' = an FAQ answer was returned although the predicted intent differs from the true intent.",
        "messages": int(len(test)), "approved_faq_answers_returned": len(answered_ids),
        "wrong_approved_answers": len(wrong_answers), "wrong_approved_answer_ids": [str(test["test_id"].iloc[i]) for i in wrong_answers],
        "fallbacks": int(len(test) - len(answered_ids)),
        "fallback_reasons": pd.Series([r.reason for r in replies if r.fallback]).value_counts().to_dict(),
        "confident_wrong_predictions": [
            {"test_id": str(test["test_id"].iloc[i]), "text": test["text"].iloc[i], "true_intent": y[i], "predicted": replies[i].intent,
             "confidence": replies[i].confidence, "outcome": replies[i].reason}
            for i in range(len(test)) if pred[i] != y[i] and conf[i] >= chat.threshold]}
    result["end_to_end"] = end_to_end
    (REPORTS / "intent_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    try:                                                           # optional image (matplotlib is not a Task 8 dependency)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("predicted intent"); ax.set_ylabel("true intent"); ax.set_title(f"Confusion matrix on chatbot_test_set (n={len(test)}, accuracy={result['accuracy']:.3f})")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout(); fig.savefig(REPORTS / "confusion_matrix.png", dpi=130); plt.close(fig)
    except ImportError:
        print("matplotlib not installed: confusion_matrix.png skipped (CSV was written)")

    print(json.dumps({k: v for k, v in result.items() if k not in ("top_confused_pairs_directed",)}, indent=2))
    print("\nper-intent F1 (lowest 10):")
    per = pd.DataFrame(rep).T.loc[labels].sort_values("f1-score")
    print(per.head(10)[["precision", "recall", "f1-score", "support"]].round(2).to_string())


if __name__ == "__main__":
    main()
