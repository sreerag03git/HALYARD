"""Platform reduced model reproduces the fitted VolturnUS-S targets."""
import numpy as np

from models.volturnus_s import build_reduced_model, VOLTURNUS_S


def test_natural_periods_match_targets():
    m = build_reduced_model()
    per = sorted(m.natural_periods_s())
    assert np.isclose(per[0], VOLTURNUS_S.pitch_natural_period_s, rtol=0.03)   # pitch ~28 s
    assert np.isclose(per[1], VOLTURNUS_S.surge_natural_period_s, rtol=0.05)   # surge ~120 s


def test_static_lean_order_published():
    m = build_reduced_model()
    x, th = m.static_response(VOLTURNUS_S.rated_thrust_ref_N)
    assert 15.0 < x < 28.0                       # ~20 m mean surge
    assert 4.0 < np.degrees(th) < 8.0            # ~5.5 deg mean pitch


def test_matrices_symmetric_positive_definite():
    m = build_reduced_model()
    assert np.allclose(m.M, m.M.T)
    assert np.allclose(m.C, m.C.T)
    assert np.all(np.linalg.eigvals(m.M) > 0)
    assert np.all(np.linalg.eigvals(m.C) > 0)
