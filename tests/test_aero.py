"""Rotor aerodynamics: calibrated surfaces hit published IEA-15MW anchors."""
import numpy as np

from models.aero_surface import load_aero_surfaces
from models.iea15mw import IEA15MW
from physics.aero import steady_operating_point, thrust_N
from physics.constants import RHO_AIR


def test_cp_max_anchor():
    # Robust to both the real IEA-15MW deck (Cp_max~0.47 @ lambda~8.5) and the calibrated
    # fallback (~0.489 @ 9); both are physical (below the Betz limit) and near the optimum.
    a = load_aero_surfaces()
    lam_opt, _, cp_max = a.optimal()
    assert 0.44 < cp_max < 0.51           # below Betz (0.593), realistic peak
    assert 8.0 < lam_opt < 9.5


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
    # MPPT-at-lambda_opt near rated wind: close to rated speed/power (the real deck's
    # lambda_opt~8.5 gives omega slightly below rated, which is acceptable below-rated running).
    tb = IEA15MW
    w0, P0 = steady_operating_point(tb, tb.rated_wind_ms)
    assert w0 > 0.72 and w0 <= tb.omega_rated_rads + 1e-6
    assert P0 > 0.9 * tb.rated_power_W


def test_steady_operating_point_below_rated_tracks_lambda_opt():
    tb = IEA15MW
    V = 8.0
    w0, P0 = steady_operating_point(tb, V)
    lam = w0 * tb.rotor_radius_m / V
    lam_opt, _, _ = tb.aero.optimal()
    assert np.isclose(lam, lam_opt, atol=0.4)
    assert P0 < tb.rated_power_W
