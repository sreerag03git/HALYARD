"""Dynamic FE cable: stability, static consistency, motion response, VIV."""
import numpy as np

from models.dynamic_cable import REFERENCE_CABLE
from physics.cable import solve_static_shape, _region_indices
from physics.cable_dynamic import solve_dynamic_cable, region_stress_from_dynamic
from physics.waves import WaveField


def _run(amp=0.3, viv=False, t_end=80.0):
    t = np.arange(0, t_end, 0.05)
    surge = amp * np.sin(2 * np.pi * t / 9)
    heave = 0.4 * amp * np.sin(2 * np.pi * t / 9 + 1)
    pitch = np.radians(amp * np.sin(2 * np.pi * t / 9))
    wave = WaveField(Hs_m=1.0, Tp_s=9.0, seed=1)
    return t, solve_dynamic_cable(REFERENCE_CABLE, t, surge, heave, pitch, wave,
                                  n_nodes=50, viv=viv)


def test_stable_and_finite():
    t, res = _run(viv=False)
    for reg in ("hang_off", "sag", "hog", "touchdown"):
        assert np.isfinite(res.region_curvature[reg]).all()
        assert np.isfinite(res.region_tension[reg]).all()


def test_mean_tension_matches_static():
    t, res = _run(amp=0.05, viv=False)
    static = solve_static_shape(REFERENCE_CABLE, n_nodes=50)
    reg = _region_indices(static, REFERENCE_CABLE)
    m = t > 20
    T_dyn = res.region_tension["hang_off"][m].mean()
    assert np.isclose(T_dyn, static.top_tension_N, rtol=0.25)   # ~static in the small-motion limit


def test_response_scales_with_motion():
    # The hang-off responds to platform pitch (via the bend stiffener); the buoyancy section
    # decouples the deep hog from fairlead motion, so use the hang-off for a clean scaling test.
    _, small = _run(amp=0.1, viv=False)
    _, big = _run(amp=0.5, viv=False)
    rng_s = np.ptp(small.region_curvature["hang_off"][40:])
    rng_b = np.ptp(big.region_curvature["hang_off"][40:])
    assert rng_b > 1.5 * rng_s     # more platform motion -> more hang-off response


def test_viv_runs_and_is_bounded():
    t, res = _run(viv=True)
    assert np.isfinite(res.region_curvature["sag"]).all()
    s = region_stress_from_dynamic(res, "hang_off")
    assert np.isfinite(s).all()
