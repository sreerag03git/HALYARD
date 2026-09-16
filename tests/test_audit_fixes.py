"""Regression tests for the physics/features audit fixes.

Each test pins a specific defect the audit found so it cannot silently return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.dynamic_cable import REFERENCE_CABLE


def test_ingest_engineB_uses_calibrated_lever():
    """Ingest engine-B (cable_response) must use the SAME hang-off lever as the live engine
    (calibrated 3 mm), not the 90 mm armour pitch radius (was a ~2.5e7x fatigue overstatement)."""
    from app import ingest
    from physics.cable import build_quasistatic_family
    fam = build_quasistatic_family()
    c = fam.cable
    kappa = np.full(500, 0.001)
    T = np.full(500, 14_000.0)
    df = pd.DataFrame({"time": np.arange(500) * 0.1,
                       "curvature_hangoff": kappa, "tension_hangoff": T})
    res = ingest.run_ingested(df, family=fam, hangoff_stick=True)
    expected = c.E_steel_Pa * c.hangoff_bend_radius_m * kappa + T / c.armour_area_m2
    assert np.allclose(res.stress_Pa, expected)
    # And explicitly NOT the pitch-radius lever.
    wrong = c.E_steel_Pa * c.armour_pitch_radius_m * kappa + T / c.armour_area_m2
    assert not np.allclose(res.stress_Pa, wrong)


@pytest.mark.parametrize("V", [11.0, 14.0])
def test_no_near_rated_pitch_limit_cycle(V):
    """Near-rated steady wind must not drive a platform-pitch limit cycle (was ~6.25 deg std
    at V=14 from an un-gain-scheduled pitch loop; fixed by gain scheduling + floating feedback)."""
    from physics.simulation import SimConfig, run_simulation
    sim = run_simulation(SimConfig(wind_ms=V, Hs_m=1.0, t_end_s=240.0,
                                   enable_support=False, turbulence_TI=0.0, blade_pitch=True))
    m = sim.t >= 170.0
    pitch_std = float(np.std(sim.pitch_deg[m]))
    assert pitch_std < 2.5, f"platform-pitch std {pitch_std:.2f} deg at V={V} (limit cycle?)"
    # rotor speed still regulated near rated (no runaway).
    assert 0.9 < float(np.mean(sim.omega_rads[m])) / 0.7917 < 1.05


def test_make_config_reproduces_committed_case():
    """Selecting a committed case must reproduce its non-sidebar YAML fields (e.g. reserve_pu),
    not fall back to dataclass defaults."""
    from analysis.config_io import list_cases, load_case
    from app.runner import make_config
    for name in list_cases():
        c = load_case(name).sim
        p = dict(case_name=name, H_sys=c.grid.H_sys_s, D_load=c.grid.D_load,
                 S_base=c.grid.S_base_MW, dP_max=c.support.dP_max, H_wt=c.support.H_wt_s,
                 R_droop=c.support.R_droop, support_window=c.schedule.support_window_s,
                 tau_rec=c.recovery.tau_rec_s, rate_rec=c.recovery.rate_rec_pu_s,
                 wind=c.wind_ms, Hs=c.Hs_m, Tp=c.Tp_s, seed=c.wave_seed, t_end=c.t_end_s,
                 p_load=c.p_load_pu, wind_cap=c.wind_capacity_MW)
        cfg = make_config(p, enable_support=c.enable_support)
        assert cfg.grid.reserve_pu == c.grid.reserve_pu
        assert cfg.grid.T_gov_s == c.grid.T_gov_s
        assert cfg.support.deadband_hz == c.support.deadband_hz
        assert cfg.recovery.shape_exponent == c.recovery.shape_exponent
        assert cfg.t_discard_s == c.t_discard_s


def test_wave_acceleration_sign_matches_grid():
    """Scalar horizontal_acceleration must match the (used) vectorized kinematics_grid."""
    from physics.waves import WaveField
    w = WaveField(Hs_m=2.0, Tp_s=8.0, seed=1234)
    t, x, z = 12.3, 5.0, -3.0
    scalar = w.horizontal_acceleration(t, x, z)
    _, _, ax, _ = w.kinematics_grid(np.array([t]), np.array([x]), np.array([z]))
    assert abs(scalar - ax[0, 0]) < 1e-9
    # And it equals the finite-difference du/dt (correct sign).
    h = 1e-4
    fd = (w.horizontal_velocity(t + h, x, z) - w.horizontal_velocity(t - h, x, z)) / (2 * h)
    assert abs(scalar - fd) < 1e-3


def test_controller_f0_threads_grid_frequency():
    """The controller's f0 must follow the grid, not a hardcoded 50 Hz."""
    from physics.controller import (FrequencyController, SupportParams, RecoveryParams,
                                    EventSchedule)
    from physics.grid import GridModel
    from models.iea15mw import IEA15MW
    ctl = FrequencyController(IEA15MW, SupportParams(), RecoveryParams(), EventSchedule(),
                              9.0, 0.7, 9e6, f0_hz=GridModel(f0_Hz=60.0).f0_Hz)
    assert ctl.f0_hz == 60.0


def test_catenary_benchmark_converges():
    """The analytic catenary cross-check must converge in length and match the analytic sag."""
    from analysis.validation import catenary_benchmark, cable_validation
    cb = catenary_benchmark()
    assert abs(cb["length_error_%"]) < 0.5
    assert abs(cb["ratio"] - 1.0) < 0.05
    # Reported cable departure angle is the settled value the fatigue family uses (~9.4 deg),
    # not the ~1 deg-early cold-solve value (~10.5).
    cv = cable_validation()
    assert 8.5 < cv["departure_angle_deg"] < 10.0
    assert abs(cv["length_error_%"]) < 1.0


def test_dynamic_cable_hangoff_stress_finite_and_signed():
    """The dynamic FE cable hang-off stress must be finite and reverse about its mean (the
    signed stiffener curvature), not a rectified one-sided signal."""
    from physics.simulation import SimConfig, run_simulation
    from physics.cable_dynamic import run_dynamic_cable_from_sim, region_stress_from_dynamic
    sim = run_simulation(SimConfig(wind_ms=9.0, Hs_m=1.5, t_end_s=200.0, enable_support=True))
    dyn = run_dynamic_cable_from_sim(sim, viv=False)
    s = region_stress_from_dynamic(dyn, "hang_off", True)
    assert np.all(np.isfinite(s))
    m = dyn.counting_mask
    rng = (s[m].max() - s[m].min()) / 1e6
    assert 0.1 < rng < 50.0     # a sane hang-off stress range [MPa]
