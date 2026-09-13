"""6-DOF BEM hydrodynamics: natural periods, passivity, head-sea symmetry, stability."""
import numpy as np
import pytest

from physics.hydro import HydroDB, build_hydro_6dof
from physics.simulation import SimConfig, run_simulation


@pytest.fixture(scope="module")
def hy():
    return build_hydro_6dof()


def test_natural_periods_match_published(hy):
    per = hy.natural_periods()
    assert np.isclose(per["surge"], 120.0, rtol=0.10)      # ~120 s
    assert np.isclose(per["heave"], 20.0, rtol=0.10)       # ~20 s
    assert np.isclose(per["pitch"], 28.0, rtol=0.10)       # ~28 s
    assert np.isclose(per["yaw"], 85.0, rtol=0.15)         # ~85 s
    assert per["sway"] == pytest.approx(per["surge"], rel=1e-6)   # symmetry


def test_retardation_passive_at_zero(hy):
    # Diagonal retardation K(0) must be >= 0 (passive body); off-diagonals unconstrained.
    assert np.all(np.diag(hy.K[0]) >= 0)


def test_added_mass_positive_definite(hy):
    Meff = hy.M + hy.A_inf
    assert np.all(np.linalg.eigvals(Meff) > 0)
    assert np.all(np.linalg.eigvals(hy.C) > 0)


def test_head_sea_symmetry():
    """Head seas (0 deg) excite surge/heave/pitch only — sway/roll/yaw stay ~0."""
    r = run_simulation(SimConfig(t_end_s=200.0, dt_s=0.05, hydro_6dof=True,
                                 wave_heading_deg=0.0))
    m = r.t >= 100
    assert np.abs(r.sway_m[m]).max() < 1e-6
    assert np.abs(r.yaw_deg[m]).max() < 1e-6
    assert r.heave_m[m].std() > 1e-3        # heave IS excited


def test_6dof_matches_2dof_qualitatively():
    """Both engines give the same grid event (nadir) — platform differs, grid does not."""
    a = run_simulation(SimConfig(t_end_s=250.0, dt_s=0.025, hydro_6dof=True))
    b = run_simulation(SimConfig(t_end_s=250.0, dt_s=0.025, hydro_6dof=False))
    assert np.isclose(a.freq_hz.min(), b.freq_hz.min(), atol=0.15)


def test_db_loads():
    db = HydroDB.load()
    assert db.A.shape[1:] == (6, 6)
    assert db.dof_order[0] == "surge"
