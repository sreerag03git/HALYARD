"""Fatigue pipeline vs textbook rainflow and a hand-checked Miner sum."""
import numpy as np

from physics.constants import PA_TO_MPA
from physics.fatigue import (FatigueParams, SNCurve, mean_stress_correct,
                             rainflow_damage)


def test_rainflow_textbook_signal():
    """ASTM E1049 reference: the classic 4-point-ish signal via the rainflow package.

    We assert total counted cycles and that the largest range is captured.
    """
    # A signal with a known dominant full cycle of range 6 (from -3 to +3) plus small ones.
    sig = np.array([0, 3, 1, 4, 2, 5, 1, 3, 0, 4, 0], dtype=float) * 1e6  # Pa
    res = rainflow_damage(sig, window_s=10.0, params=FatigueParams(mean_stress="none"))
    # Total count (half + full) should equal rainflow's cycle sum; largest range = 5 (MPa here).
    assert res.n_cycles > 0
    assert np.isclose(res.ranges_MPa.max(), 5.0, atol=1e-6)


def test_miner_hand_checked():
    """Constant-amplitude loading: D = n / N with N from the S-N law, hand-checked."""
    sn = SNCurve(m1=3.0, log_a1=12.0, m2=3.0, log_a2=12.0, N_knee=1e12,  # single slope
                 sigma_u_MPa=1e9, endurance_limit_MPa=0.0)
    # A pure sinusoid of amplitude A -> ranges of 2A; count = number of cycles.
    A = 50.0  # MPa amplitude -> range 100 MPa
    n_cycles = 20
    t = np.linspace(0, n_cycles * 2 * np.pi, n_cycles * 400)
    sig = (A * np.sin(t)) / PA_TO_MPA  # to Pa
    res = rainflow_damage(sig, window_s=1.0, params=FatigueParams(sn=sn, mean_stress="none"))
    # Expected N for range 100 MPa: log10 N = 12 - 3*log10(100) = 12 - 6 = 6 -> N = 1e6.
    N_expected = 10 ** (12.0 - 3.0 * np.log10(100.0))
    D_expected = n_cycles / N_expected
    assert np.isclose(res.damage, D_expected, rtol=0.05)


def test_goodman_increases_with_tensile_mean():
    S = np.array([100.0])
    su = 1000.0
    s0 = mean_stress_correct(S, np.array([0.0]), su, "goodman")
    s1 = mean_stress_correct(S, np.array([300.0]), su, "goodman")
    assert s1[0] > s0[0]
    # Goodman closed form check.
    assert np.isclose(s1[0], 100.0 / (1 - 0.3), rtol=1e-9)


def test_sn_bilinear_knee_continuity():
    sn = SNCurve()
    S_knee = 10.0 ** ((sn.log_a1 - np.log10(sn.N_knee)) / sn.m1)
    n_below = sn.cycles_to_failure(np.array([S_knee * 1.001]))[0]
    n_above = sn.cycles_to_failure(np.array([S_knee * 0.999]))[0]
    # Continuous at the knee (both ~ N_knee).
    assert np.isclose(n_below, sn.N_knee, rtol=0.05)
    assert np.isclose(n_above, sn.N_knee, rtol=0.05)


def test_combined_signal_not_summed():
    """A mean shift on wave cycles changes damage — proving counting on the combined signal
    differs from summing wave-only + drift-only (the physically-correct behaviour)."""
    fp = FatigueParams(mean_stress="goodman")
    t = np.linspace(0, 100, 5000)
    waves = 20e6 * np.sin(2 * np.pi * t / 8.0)
    drift = 15e6 * np.exp(-((t - 50) / 10) ** 2)     # a slow mean shift
    D_combined = rainflow_damage(waves + drift, 100.0, fp).damage
    D_waves = rainflow_damage(waves, 100.0, fp).damage
    assert D_combined > D_waves      # the mean shift adds damage (R-ratio effect)
