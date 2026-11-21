import json
import warnings
from pathlib import Path
import pytest

warnings.filterwarnings('ignore')

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

with open(FIXTURES_DIR / "real_samples.json", "r", encoding="utf-8") as f:
    FIX = json.load(f)


def real(name: str):
    return FIX[name]["body"]
