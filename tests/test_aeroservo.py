"""Aero-servo: turbulent inflow, blade-pitch control (above-rated), dynamic inflow."""
import numpy as np

from physics.simulation import SimConfig, run_simulation
from physics.wind import TurbulentWind


def test_kaimal_turbulence_intensity():
    tw = TurbulentWind(U_mean=10.0, TI=0.12, seed=1)
    t = np.arange(0, 600, 0.1)
    ti = tw.realized_TI(t)
    assert 0.06 < ti < 0.14        # realized TI near the target (finite band -> a bit low)
    assert (tw.series(t) > 0).all()


def test_blade_pitch_holds_speed_above_rated():
    r = run_simulation(SimConfig(t_end_s=200.0, dt_s=0.025, wind_ms=15.0, blade_pitch=True))
    m = r.t > 60
    assert r.meta["beta0_deg"] > 3.0               # pitched to shed load
    assert 0.72 < r.omega_rads[m].mean() < 0.82    # held near rated speed
    assert np.isfinite(r.surge_m).all()


def test_below_rated_pitch_is_zero():
    r = run_simulation(SimConfig(t_end_s=150.0, dt_s=0.025, wind_ms=8.0, blade_pitch=True))
    assert np.abs(r.blade_pitch_deg).max() < 0.5   # no pitch below rated


def test_turbulence_increases_platform_motion():
    calm = run_simulation(SimConfig(t_end_s=200.0, dt_s=0.025, wind_ms=9.0, turbulence_TI=0.0))
    turb = run_simulation(SimConfig(t_end_s=200.0, dt_s=0.025, wind_ms=9.0, turbulence_TI=0.12))
    m = calm.t > 60
    assert turb.surge_m[m].std() > 3 * calm.surge_m[m].std()
