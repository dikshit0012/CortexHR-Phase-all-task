"""Build src/cortexhr_task9/typo_corrections.json from the TRAINING file only.

Rule (deliberately narrow): a token is a typo of a more frequent token when it differs from it by ONE adjacent-letter swap
('laeve' -> 'leave') and is rarer. Everything produced here was reviewed by hand; REVIEWED_EXCLUDE lists real words that
the rule would wrongly rewrite, REVIEWED_ADD lists typos the rule cannot find. The JSON file is static at run time, so
training and inference use exactly the same corrections.
Run from the `task 9` folder:  python scripts/build_typo_map.py
"""
import collections
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "chatbot_nlu_training.csv"
OUT = ROOT / "src" / "cortexhr_task9" / "typo_corrections.json"

REVIEWED_EXCLUDE = {"form"}              # 'form' -> 'from' would corrupt 'Form 16' (tax_declaration)
REVIEWED_ADD = {"makrs": "marks"}        # 'how many late makrs do I have' - correct form is not in the data
MIN_LEN = 4
MAX_TYPO_COUNT = 6


def swaps(word):
    return {word[:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(len(word) - 1) if word[i] != word[i + 1]}


def main():
    train = pd.read_csv(TRAIN, keep_default_na=False)
    counts = collections.Counter(t for s in train["text"] for t in re.findall(r"[a-z0-9&']+", s.lower()))
    found = {}
    for word, c in counts.items():
        if len(word) < MIN_LEN:
            continue
        for variant in swaps(word):
            if variant in counts and counts[variant] < c and counts[variant] <= MAX_TYPO_COUNT:
                found[variant] = word
    for bad in REVIEWED_EXCLUDE:
        found.pop(bad, None)
    found.update(REVIEWED_ADD)
    OUT.write_text(json.dumps(dict(sorted(found.items())), indent=1), encoding="utf-8")
    print(f"wrote {len(found)} corrections to {OUT.name}")


if __name__ == "__main__":
    main()
