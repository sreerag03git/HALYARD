"""Rotor aerodynamics: calibrated surfaces hit published IEA-15MW anchors."""
import numpy as np

from models.aero_surface import load_aero_surfaces
from models.iea15mw import IEA15MW
from physics.aero import steady_operating_point, thrust_N
from physics.constants import RHO_AIR


def test_cp_max_anchor():
    a = load_aero_surfaces()
    lam_opt, _, cp_max = a.optimal()
    assert np.isclose(cp_max, 0.489, atol=0.01)
    assert np.isclose(lam_opt, 9.0, atol=0.3)


def test_rated_thrust_order():
    tb = IEA15MW
    omega = tb.omega_rated_rads
    T = thrust_N(tb, omega, tb.rated_wind_ms, 0.0)
    assert 2.2e6 < T < 2.7e6      # ~2.4 MN published-order rated thrust


def test_ct_slope_positive_in_operating_band():
    """dC_T/dlambda > 0 for 7 < lambda < 10 — the sign that makes rotor slowdown shed thrust."""
    a = load_aero_surfaces()
    for lam in (7.0, 8.0, 9.0):
        d = (a.ct(lam + 0.01, 0) - a.ct(lam - 0.01, 0)) / 0.02
        assert d > 0


def test_steady_operating_point_rated():
    tb = IEA15MW
    w0, P0 = steady_operating_point(tb, tb.rated_wind_ms)
    assert np.isclose(w0, tb.omega_rated_rads, rtol=0.02)
    assert np.isclose(P0, tb.rated_power_W, rtol=0.02)


def test_steady_operating_point_below_rated_tracks_lambda_opt():
    tb = IEA15MW
    V = 8.0
    w0, P0 = steady_operating_point(tb, V)
    lam = w0 * tb.rotor_radius_m / V
    lam_opt, _, _ = tb.aero.optimal()
    assert np.isclose(lam, lam_opt, atol=0.3)
    assert P0 < tb.rated_power_W
