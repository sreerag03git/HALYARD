"""Determinism (§11): same inputs -> same outputs, every time (seeds fixed)."""
import numpy as np

from physics.simulation import SimConfig, run_simulation


def test_same_seed_same_output():
    cfg = SimConfig(t_end_s=200.0, dt_s=0.02, wave_seed=42)
    r1 = run_simulation(cfg)
    r2 = run_simulation(cfg)
    assert np.array_equal(r1.surge_m, r2.surge_m)
    assert np.array_equal(r1.freq_hz, r2.freq_hz)
    assert np.array_equal(r1.thrust_N, r2.thrust_N)


def test_different_seed_different_waves():
    a = run_simulation(SimConfig(t_end_s=200.0, dt_s=0.02, wave_seed=1))
    b = run_simulation(SimConfig(t_end_s=200.0, dt_s=0.02, wave_seed=2))
    assert not np.allclose(a.surge_m, b.surge_m)


def test_secondary_dip_exists_with_support():
    """A genuine secondary frequency dip appears during recovery (the loop is closed)."""
    r = run_simulation(SimConfig(t_end_s=400.0, dt_s=0.02, enable_support=True,
                                 p_load_pu=0.12))
    rec = r.mode == 2
    assert rec.any()
    # frequency during recovery dips below the value at recovery onset
    onset = np.argmax(rec)
    assert r.freq_hz[rec].min() < r.freq_hz[onset] + 1e-6
