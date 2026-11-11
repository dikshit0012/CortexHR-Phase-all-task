"""Diagnostic: send each of the 30 official FAQ questions through the full chat pipeline (saved model + threshold + FAQ rules).

Not a tuning tool and not the official test set - it answers 'does the bot handle the FAQ's own wording?'.
Writes reports/faq_question_self_check.csv.   Run from the `task 9` folder:   python scripts/check_faq_questions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cortexhr_task9.chat_service import load_chat_service  # noqa: E402
from cortexhr_task9.paths import REPORTS  # noqa: E402


def main() -> None:
    svc = load_chat_service()
    reachable = set(svc.responder.reachable_faq_ids())
    rows = []
    for fid, e in sorted(svc.responder.faq.items()):
        r = svc.reply(e.question)
        if not r.fallback:
            outcome = "answered_correctly" if r.faq_ids == (fid,) else "ANSWERED_WITH_A_DIFFERENT_FAQ"
        elif fid not in reachable:
            outcome = "fallback (no intent covers this FAQ)"
        elif r.reason == "low_confidence":
            outcome = "fallback (confidence below threshold)"
        else:
            outcome = f"fallback ({r.reason})"
        rows.append({"faq_id": fid, "question": e.question, "predicted_intent": r.intent, "confidence": r.confidence,
                     "outcome": outcome, "answered_with": ",".join(r.faq_ids)})
    df = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / "faq_question_self_check.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 50)
    print(df.to_string(index=False)); print("\n", df["outcome"].value_counts().to_string())


if __name__ == "__main__":
    main()
