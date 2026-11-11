"""Adapter around the Task 9 intent model. Prediction only: no answers, no service calls, no authorization."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float


class IntentClassifier(Protocol):
    threshold: float

    def classify(self, text: str) -> Prediction: ...


class Task9Classifier:
    """Reuses the saved Task 9 pipeline (TF-IDF + Logistic Regression) and its confidence threshold unchanged."""

    def __init__(self, chat_service: Any):
        self._model = chat_service.model
        self.threshold = float(chat_service.threshold)

    @classmethod
    def load(cls, task9_root: Optional[Path] = None) -> "Task9Classifier":
        root = Path(task9_root or os.environ.get("CORTEXHR_TASK9_ROOT") or
                    Path(__file__).resolve().parents[3] / "task09" / "cortexhr_task9" / "task 9")
        src = str(root / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from cortexhr_task9.chat_service import load_chat_service      # noqa: WPS433 - late import, optional component
        return cls(load_chat_service())

    def classify(self, text: str) -> Prediction:
        p = self._model.predict(text)
        return Prediction(p.intent, float(p.confidence))
