"""Validate the intent -> FAQ mapping against the TRAINING messages (true intents; no model involved, test set not used).

For every distinct base phrasing of every intent that has FAQ rules it shows which approved FAQ answer the rules select, or
'(no approved answer)'. Use it to review the mapping by eye. Writes reports/faq_mapping_validation.csv and reports/faq_coverage.md.
Run from the `task 9` folder:   python scripts/check_faq_coverage.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cortexhr_task9.data import load_training, phrase_group  # noqa: E402
from cortexhr_task9.faq import load_responder  # noqa: E402
from cortexhr_task9.paths import DEFAULT_FAQ, INTENT_FAQ_MAP, REPORTS, TRAIN_CSV  # noqa: E402


def main() -> None:
    train = load_training(TRAIN_CSV)
    intents = sorted(train["intent"].unique())
    resp = load_responder(DEFAULT_FAQ, INTENT_FAQ_MAP, intents)
    train["phrase"] = train["text"].map(phrase_group)
    rows = []
    for intent, g in train.groupby("intent"):
        for phrase, gg in g.groupby("phrase"):
            ans = resp.answer(intent, phrase)
            rows.append({"intent": intent, "base_phrasing": phrase, "messages": len(gg),
                         "faq_ids": ",".join(ans.faq_ids) if ans else "", "rule": ans.rule if ans else "(no approved answer)"})
    df = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / "faq_mapping_validation.csv", index=False)

    per = df.groupby("intent").apply(lambda d: pd.Series({
        "has_faq_rules": resp.has_rules(d.name), "base_phrasings": len(d),
        "phrasings_answered": int((d["faq_ids"] != "").sum()),
        "messages": int(d["messages"].sum()), "messages_answered": int(d.loc[d["faq_ids"] != "", "messages"].sum())}), include_groups=False)
    reach = set(resp.reachable_faq_ids())
    all_ids = sorted(resp.faq)
    unreachable = [i for i in all_ids if i not in reach]
    lines = ["# FAQ mapping validation (training messages, true intents)", "",
             "| intent | FAQ rules? | phrasings answered | messages answered |", "|---|---|---|---|"]
    for intent, r in per.iterrows():
        lines.append(f"| {intent} | {'yes' if r.has_faq_rules else 'no'} | {int(r.phrasings_answered)}/{int(r.base_phrasings)} | {int(r.messages_answered)}/{int(r.messages)} |")
    lines += ["", f"FAQ entries reachable through some intent: {len(reach)} of {len(all_ids)}.", "",
              "FAQ entries NOT reachable (no intent in the training data covers them): " + ", ".join(f"{i} ({resp.faq[i].question})" for i in unreachable), ""]
    (REPORTS / "faq_coverage.md").write_text("\n".join(lines), encoding="utf-8")
    print(per.to_string()); print("\nunreachable:", unreachable)
    pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 500)
    print("\n", df[df["intent"].isin([i for i in intents if resp.has_rules(i)])][["intent", "base_phrasing", "faq_ids", "rule"]].to_string(index=False))


if __name__ == "__main__":
    main()
