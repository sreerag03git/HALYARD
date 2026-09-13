"""Metocean scatter and annualization sanity checks."""
import numpy as np

from models.metocean import SCATTER, check_occurrence_sums_to_one


def test_occurrence_sums_to_one():
    assert np.isclose(check_occurrence_sums_to_one(), 1.0, atol=1e-9)


def test_bins_are_ordered_and_reasonable():
    hs = [s.Hs_m for s in SCATTER]
    assert hs == sorted(hs)
    assert 5 <= len(SCATTER) <= 10          # a reduced, fatigue-relevant set
    assert all(0.5 <= s.Hs_m <= 8.0 and 4.0 <= s.Tp_s <= 16.0 for s in SCATTER)
