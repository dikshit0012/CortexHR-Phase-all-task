"""Train the HR intent classifier on chatbot_nlu_training.csv ONLY and save the versioned artifact.

Run from the `task 9` folder:   python scripts/train_intent_model.py
Writes: models/hr_intent_classifier.joblib (+ .metadata.json) and dev-split evidence in reports/.
The official test set (chatbot_test_set.csv) is NOT read by this script.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortexhr_task9.paths import DEFAULT_MODEL, REPORTS, TRAIN_CSV, env_path, ENV_MODEL  # noqa: E402
from cortexhr_task9.training import train_and_save  # noqa: E402


def main() -> None:
    model_path = env_path(ENV_MODEL, DEFAULT_MODEL)
    out = train_and_save(TRAIN_CSV, model_path, REPORTS)
    meta = out["metadata"]
    print(f"saved {model_path.name}  ({meta['training']['rows']} rows, {meta['training']['intents']} intents, scikit-learn {meta['sklearn_version']})")
    print("selected threshold:", meta["threshold"])
    print(json.dumps(meta["threshold_selection"], indent=2))
    print("\nconfig comparison (grouped CV on training data):")
    print(out["config_comparison"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
