"""DLC fatigue matrix + multi-seed UQ."""
import numpy as np
import pytest

from analysis.config_io import load_case
from analysis.dlc import run_dlc, wind_weibull_weights, sea_state_for_wind
from physics.cable import build_quasistatic_family


def test_weibull_weights_normalized():
    w = wind_weibull_weights([6, 9, 12, 15, 18])
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()


def test_sea_state_monotonic_with_wind():
    hs = [sea_state_for_wind(v)[0] for v in (6, 10, 14, 18)]
    assert hs == sorted(hs)          # Hs increases with wind


def test_dlc_runs_and_bands_ordered():
    fam = build_quasistatic_family()
    c = load_case("default")
    r = run_dlc(c.sim, events_per_year=c.events_per_year, winds=(9.0, 14.0),
                headings=(0.0,), seeds=(1234, 5678), family=fam, fatigue_params=c.fatigue,
                daf=c.daf, hangoff_stick=c.hangoff_stick, t_end=200.0)
    assert np.isfinite(r.life_years_mean) and r.life_years_mean > 0
    assert r.life_years_lo <= r.life_years_mean <= r.life_years_hi
    assert 0.0 <= r.control_share_pct <= 100.0
