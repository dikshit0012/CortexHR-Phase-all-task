import math
from pathlib import Path

import numpy as np
import pytest

from monitoring import monitoring_config as cfg
from monitoring.psi import PSIError, classify_psi, compute_psi, psi_categorical, psi_numeric

HELPER = Path(__file__).resolve().parents[3] / "phase-3" / "task15" / "src" / "cortexhr_task15" / "cortexhr_helpers.py"
RNG = np.random.default_rng(7)


def original_psi():
    src = HELPER.read_text()
    ns = {}
    exec("import numpy as np\n" + src[src.index("def psi"):], ns)       # the course helper, taken verbatim from the repo
    return ns["psi"]


def manual(ref_counts, cur_counts, eps=1e-6):
    pr, pc = np.array(ref_counts, float), np.array(cur_counts, float)
    pr, pc = np.clip(pr / pr.sum(), eps, None), np.clip(pc / pc.sum(), eps, None)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def test_numeric_matches_the_original_helper_on_continuous_data():
    old = original_psi()
    a = RNG.normal(40, 10, 3000)
    for cur in (a, RNG.normal(43, 10, 1000), RNG.normal(50, 12, 500), a + 1000):
        assert psi_numeric(a, cur).psi == pytest.approx(old(a, cur), abs=1e-6)


def test_exact_value_against_hand_calculation_categorical():
    ref = [1] * 10 + [2] * 20 + [3] * 40 + [4] * 30
    cur = [1] * 10 + [2] * 20 + [3] * 20 + [4] * 50
    assert psi_categorical(ref, cur).psi == pytest.approx(manual([10, 20, 40, 30], [10, 20, 20, 50]), abs=1e-6)
    assert psi_numeric(ref, cur).psi == pytest.approx(manual([10, 20, 40, 30], [10, 20, 20, 50]), abs=1e-6)   # <=10 distinct -> per level


def test_categorical_strings_and_unseen_category():
    ref = ["Hybrid"] * 50 + ["Onsite"] * 30 + ["Remote"] * 20
    assert psi_categorical(ref, ref).psi == 0
    r = psi_categorical(ref, ref + ["Satellite"] * 40)                    # a brand-new category must raise PSI, not be ignored
    assert r.psi > cfg.PSI_ALERT and r.top_bin == "Satellite"
    assert psi_categorical([1, 2, 3.0, "3 "], [1, 2, 3, 3]).psi == 0       # 3, 3.0, '3 ' are one level


def test_binary_shift_is_detected_where_the_original_helper_returned_zero():
    ref, cur = RNG.binomial(1, .3, 2000), RNG.binomial(1, .6, 2000)
    assert original_psi()(ref, cur) == 0.0                                # documents the audited defect
    assert classify_psi(psi_numeric(ref, cur).psi) == "ALERT"


def test_missing_values_form_their_own_bin():
    a = RNG.normal(0, 1, 2000)
    with_nan = np.r_[a, np.full(500, np.nan)]
    r = psi_numeric(with_nan, a)
    assert r.reference_missing == 500 and r.current_missing == 0 and r.psi > cfg.PSI_ALERT     # missing rate jump = drift
    same = psi_numeric(with_nan, np.r_[a, np.full(500, np.nan)])
    assert same.psi < 0.01
    assert psi_categorical(["a", None, np.nan, ""], ["a", "a"]).reference_missing == 3


def test_zero_frequency_bins_are_protected():
    ref = RNG.normal(0, 1, 2000)
    cur = np.full(100, ref.max() + 5)                                     # every current value lands in the top bin: other bins have 0%
    r = psi_numeric(ref, cur)
    assert math.isfinite(r.psi) and r.psi > 0
    assert math.isfinite(psi_categorical(["a", "b"], ["a", "a"]).psi)


def test_identical_distributions_are_zero_and_different_are_not():
    a = RNG.normal(0, 1, 2000)
    assert psi_numeric(a, a).psi == 0
    assert classify_psi(psi_numeric(a, RNG.normal(0, 1, 800)).psi) == "OK"        # sampling noise only
    assert classify_psi(psi_numeric(a, RNG.normal(1, 1, 800)).psi) == "ALERT"


@pytest.mark.parametrize("value,expected", [(0.0, "OK"), (0.099999, "OK"), (0.10, "WATCH"), (0.100001, "WATCH"), (0.18, "WATCH"), (0.25, "WATCH"),
                                            (0.250001, "ALERT"), (3.0, "ALERT"), (float("nan"), "NOT_EVALUATED"), (None, "NOT_EVALUATED"),
                                            (float("inf"), "NOT_EVALUATED")])
def test_threshold_boundaries(value, expected):
    assert classify_psi(value) == expected


@pytest.mark.parametrize("bad", [
    lambda: psi_numeric([1, 2, 3], []), lambda: psi_numeric([], [1]), lambda: psi_numeric(["x"] * 30, [1.0] * 30),
    lambda: psi_numeric([np.inf] * 20 + list(range(20)), list(range(20))), lambda: psi_numeric([np.nan] * 20, list(range(20))),
    lambda: psi_categorical([], ["a"]), lambda: compute_psi([1], [1], "ordinal"),
])
def test_invalid_input_raises_instead_of_returning_zero(bad):
    with pytest.raises(PSIError):
        bad()


def test_reference_too_concentrated_raises():
    ref = np.r_[np.zeros(990), np.arange(1, 12)]                          # 12 distinct values, but 99% identical: quantile edges collapse
    with pytest.raises(PSIError):
        psi_numeric(ref, ref)


def test_result_reports_sizes_and_top_bin():
    r = psi_numeric(RNG.normal(0, 1, 500), RNG.normal(2, 1, 300))
    assert (r.reference_n, r.current_n, r.method) == (500, 300, "numeric_quantile") and r.top_bin and r.n_bins >= 3


def test_eight_features_configured_and_unique():
    names = [f.name for f in cfg.FEATURES]
    assert len(names) == len(set(names)) == 8 and {f.kind for f in cfg.FEATURES} <= {"numeric", "categorical"}
