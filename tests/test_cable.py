"""Cable static solve: inextensibility, lazy-wave shape, analytic catenary cross-check."""
import dataclasses

import numpy as np

from models.dynamic_cable import REFERENCE_CABLE
from physics.cable import (build_quasistatic_family, region_stress_timeseries,
                           solve_static_shape, _region_indices)


def test_length_conservation():
    # Suspended segments are exactly inextensible (Gauss-Seidel); the seabed-laid tail is
    # straightened as a modelling choice, so total length is within a few percent.
    sh = solve_static_shape(REFERENCE_CABLE)
    seg = np.sqrt(np.diff(sh.x) ** 2 + np.diff(sh.z) ** 2)
    assert np.isclose(seg.sum(), REFERENCE_CABLE.total_length_m, rtol=0.03)


def test_lazy_wave_has_hog_above_neighbours():
    sh = solve_static_shape(REFERENCE_CABLE)
    reg = _region_indices(sh, REFERENCE_CABLE)
    # The hog (buoyancy crest) sits above the sag trough.
    assert sh.z[reg["hog"]] > sh.z[reg["sag"]]
    # Touchdown is on the seabed.
    assert sh.z[reg["touchdown"]] < -REFERENCE_CABLE.water_depth_m + 3.0


def test_mbr_respected():
    sh = solve_static_shape(REFERENCE_CABLE)
    assert sh.curvature.max() < REFERENCE_CABLE.curvature_limit_1_per_m


def test_catenary_sag_matches_closed_form():
    """Fully-suspended symmetric bare catenary: solved sag ~ analytic a(cosh(c/a)-1)."""
    from analysis.validation import catenary_benchmark
    b = catenary_benchmark()
    assert 0.85 < b["ratio"] < 1.15      # within 15% of the closed-form sag


def test_quasistatic_family_smooth_and_monotone_angle():
    fam = build_quasistatic_family(REFERENCE_CABLE, n=9)
    # Departure angle should vary smoothly and monotonically with the top offset.
    d = np.diff(fam.angle_deg)
    assert np.all(d < 0) or np.all(d > 0)
    assert np.max(np.abs(np.diff(d))) < 0.02      # low curvature -> smooth


def test_hangoff_stress_responds_to_pitch():
    fam = build_quasistatic_family(REFERENCE_CABLE, n=9)
    t = np.linspace(0, 100, 2000)
    surge = 0.2 * np.sin(2 * np.pi * t / 9)
    pitch_small = np.radians(0.1 * np.sin(2 * np.pi * t / 9))
    pitch_big = np.radians(0.5 * np.sin(2 * np.pi * t / 9))
    s_small, _, _ = region_stress_timeseries(fam, "hang_off", surge, pitch_small)
    s_big, _, _ = region_stress_timeseries(fam, "hang_off", surge, pitch_big)
    assert np.ptp(s_big) > np.ptp(s_small)        # more pitch -> more hang-off stress
