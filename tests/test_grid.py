"""Grid SFR model vs the analytical swing-equation step response."""
import numpy as np

from physics.grid import GridModel, IDX_DF, IDX_PGOV, IDX_ROCOF, N_GRID_STATES


def _integrate(grid, p_load, t_end=120.0, dt=0.005):
    y = grid.initial_state()
    n = int(t_end / dt)
    for _ in range(n):
        d, _ = grid.rhs(y, 0.0, p_load)
        y = y + dt * d
    return y


def test_initial_rocof_matches_swing_equation():
    """At the instant of a load step, RoCoF = -dP*f0/(2H) (governor still zero)."""
    g = GridModel(H_sys_s=3.0)
    dP = 0.10
    d, rocof_true = g.rhs(g.initial_state(), 0.0, dP)
    expected = -dP * g.f0_Hz / (2.0 * g.H_sys_s)
    assert np.isclose(rocof_true, expected, rtol=1e-6)
    assert np.isclose(d[IDX_DF], -dP / (2.0 * g.H_sys_s), rtol=1e-6)


def test_steady_state_frequency_deviation():
    """With primary response (reserve not exhausted): df_ss = -dP / (1/R + D)."""
    g = GridModel(H_sys_s=3.0, D_load=1.0, R_sys=0.05, reserve_pu=0.5, T_gov_s=3.0)
    dP = 0.02
    y = _integrate(g, dP, t_end=300.0, dt=0.005)
    df_ss = y[IDX_DF]
    expected = -dP / (1.0 / g.R_sys + g.D_load)
    assert np.isclose(df_ss, expected, rtol=0.02)


def test_lower_inertia_gives_steeper_rocof():
    weak = GridModel(H_sys_s=1.5)
    strong = GridModel(H_sys_s=6.0)
    _, r_weak = weak.rhs(weak.initial_state(), 0.0, 0.1)
    _, r_strong = strong.rhs(strong.initial_state(), 0.0, 0.1)
    assert abs(r_weak) > abs(r_strong)


def test_state_dimensions():
    g = GridModel()
    assert g.initial_state().shape == (N_GRID_STATES,)
