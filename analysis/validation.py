"""Validation & sensitivity support (§10).

Surfaces the checks a reviewer expects:
  * platform reduced model reproduces published VolturnUS-S natural periods and static lean;
  * JONSWAP realization reproduces the target Hs (4 sqrt(m0));
  * cable static solve conserves length (inextensibility), balances top tension against the
    suspended weight, and respects the MBR — plus an analytic catenary cross-check for a
    bare (no-buoyancy) cable.
The sensitivity story (S-N slope m, Goodman/Gerber) is exercised live in the app; this
module provides the numbers behind the validation tab.
"""
from __future__ import annotations

import numpy as np

from models.dynamic_cable import DynamicCable, REFERENCE_CABLE
from models.volturnus_s import build_reduced_model
from physics.cable import solve_static_shape
from physics.waves import WaveField


def platform_validation() -> dict:
    m = build_reduced_model()
    fr = m.fit_report
    return {
        "target_periods_s": {"surge": fr["target_surge_period_s"],
                             "pitch": fr["target_pitch_period_s"]},
        "fitted_periods_s": sorted(fr["fitted_natural_periods_s"]),
        "static_surge_at_rated_m": fr["static_surge_at_rated_m"],
        "static_pitch_at_rated_deg": fr["static_pitch_at_rated_deg"],
        "added_mass_fraction_surge": fr["added_mass_fraction_surge"],
    }


def wave_validation(Hs_list=(1.0, 2.5, 4.0), Tp=8.0, seed=1234) -> list:
    out = []
    for Hs in Hs_list:
        w = WaveField(Hs_m=Hs, Tp_s=Tp, seed=seed)
        out.append({"Hs_target_m": Hs, "Hs_realized_m": w.Hs_realized,
                    "error_%": 100 * (w.Hs_realized - Hs) / Hs})
    return out


def cable_validation(cable: DynamicCable | None = None) -> dict:
    cable = cable or REFERENCE_CABLE
    sh = solve_static_shape(cable)
    # Length conservation (inextensibility).
    seg = np.sqrt(np.diff(sh.x) ** 2 + np.diff(sh.z) ** 2)
    solved_len = float(seg.sum())
    # Top tension vs suspended-weight balance (order-of-magnitude force check).
    td = sh.touchdown_idx
    suspended_w = cable.submerged_weight_N_per_m * sh.s[td]   # crude vertical load
    return {
        "converged": sh.converged,
        "solved_length_m": solved_len,
        "target_length_m": cable.total_length_m,
        "length_error_%": 100 * (solved_len - cable.total_length_m) / cable.total_length_m,
        "top_tension_kN": sh.top_tension_N / 1e3,
        "suspended_weight_kN": suspended_w / 1e3,
        "max_curvature_1_per_m": float(sh.curvature.max()),
        "mbr_limit_1_per_m": cable.curvature_limit_1_per_m,
        "mbr_ok": bool(sh.curvature.max() < cable.curvature_limit_1_per_m),
        "departure_angle_deg": sh.departure_angle_deg,
    }


def catenary_benchmark(cable: DynamicCable | None = None) -> dict:
    """Closed-form cross-check: a fully-suspended symmetric bare catenary.

    A bare cable (no buoyancy) hung between two equal-height points, clear of the seabed,
    is a classic catenary. For span 2c and suspended length 2L, the catenary parameter a
    solves L = a sinh(c/a), and the mid-span sag is d = a(cosh(c/a) - 1). We solve the same
    case numerically and compare the sag — a geometric quantity that validates the solver's
    force balance while being tolerant of the small bending stiffness (which only slightly
    reduces the peak curvature, not the overall sag).
    """
    import dataclasses
    from scipy.optimize import brentq
    cable = cable or REFERENCE_CABLE
    bare = dataclasses.replace(cable, buoyancy_uplift_factor=0.0)
    span, half_len, z0 = 120.0, 75.0, -8.0    # 2c=120 span, 2L=150 length, deep (no seabed)
    bench = dataclasses.replace(bare, total_length_m=2 * half_len, water_depth_m=400.0,
                                horizontal_layout_m=span)
    sh = solve_static_shape(bench, x_top=0.0, z_top=z0, anchor_x=span, anchor_z=z0,
                            n_nodes=80, smooth_w=0.03)
    sag_numeric = z0 - float(sh.z.min())
    c = span / 2.0
    a = brentq(lambda a_: a_ * np.sinh(c / a_) - half_len, 1.0, 1e5)
    sag_analytic = a * (np.cosh(c / a) - 1.0)
    return {"span_m": span, "length_m": 2 * half_len,
            "catenary_a_m": a, "sag_analytic_m": sag_analytic,
            "sag_numeric_m": sag_numeric,
            "ratio": sag_numeric / sag_analytic if sag_analytic else np.nan}
