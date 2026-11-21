import warnings
from pathlib import Path
import pandas as pd
import synthetic

warnings.filterwarnings('ignore')

# Fixture to create synthetic reference and current CSVs for drift monitoring tests
import pytest

@pytest.fixture
def drift_csvs(tmp_path, monkeypatch):
    """Create reference and current CSV files using the synthetic data generators.
    The returned callable allows the test to customise the datasets.
    """
    def _make(
        n_ref: int = 2000,
        n_cur: int = 1000,
        ext_kwargs: dict | None = None,
        cur_kwargs: dict | None = None,
    ):
        # Reference data – historically extended plus optional matched snapshot rows
        ref_df = pd.concat([
            synthetic.attrition_frame(n_ref, seed=1, source="historical_extension", **(ext_kwargs or {})),
        ], ignore_index=True)
        ref_path = tmp_path / "ref.csv"
        ref_df.to_csv(ref_path, index=False)
        # Current scoring data
        cur_df = synthetic.attrition_frame(n_cur, seed=2, **(cur_kwargs or {}))
        cur_path = tmp_path / "cur.csv"
        cur_df.to_csv(cur_path, index=False)
        # Export environment variables for the monitor to locate the files
        monkeypatch.setenv("CORTEXHR_REFERENCE_CSV", str(ref_path))
        monkeypatch.setenv("CORTEXHR_SCORING_CSV", str(cur_path))
        return {
            "reference": ref_path,
            "current": cur_path,
        }
    return _make
