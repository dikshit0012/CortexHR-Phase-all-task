import pandas as pd

from cortexhr.audit import iqr_mask, normalise_location, z_mask


def test_iqr_flags_the_extreme_value():
    s = pd.Series([10, 11, 12, 11, 10, 12, 11, 500])
    assert iqr_mask(s).tolist() == [False] * 7 + [True]


def test_z_mask_needs_a_big_sample_to_fire():
    s = pd.Series([10] * 50 + [100])
    assert z_mask(s).iloc[-1]
    assert not z_mask(s).iloc[:-1].any()


def test_iqr_overflags_zero_inflated_data_more_than_z():
    # 80% zeros -> IQR is 0 so ANY non-zero is flagged; that is a known weakness of IQR.
    s = pd.Series([0] * 80 + [1, 2, 3, 4, 5] * 4)
    assert iqr_mask(s).sum() == 20
    assert z_mask(s).sum() < iqr_mask(s).sum()   # z-score is far less trigger-happy here


def test_location_aliases():
    for raw, want in [("BENGALURU", "Bengaluru"), ("Bangalore", "Bengaluru"), ("Madras", "Chennai"),
                      ("delhi ncr", "Delhi NCR"), ("Delhi-NCR", "Delhi NCR"), ("chennai", "Chennai"),
                      ("Pune", "Pune")]:
        assert normalise_location(raw) == want
