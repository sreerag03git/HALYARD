"""UMaine VolturnUS-S semi-submersible — frozen reference values + reduced-order fit.

Source: Allen et al., "Definition of the UMaine VolturnUS-S Reference Platform
Developed for the IEA Wind 15-Megawatt Offshore Reference Wind Turbine",
NREL/TP-5000-76773 (2020). Values below are read-only reference data.

Reduced-order motion model (§5.5)
---------------------------------
HALYARD's live engine integrates a 2-DOF (surge x, pitch theta) linear model. The
spec permits fitting the reduced coefficients to published dynamics; we do so
*transparently* by anchoring the two quantities that actually govern the fatigue
mechanism, then verifying the fit (validation tab, §10):

  1. **Static lean per unit thrust** — the slow, control-driven excursion that
     shifts the mean stress at the hang-off is set by the restoring stiffness.
     We anchor surge stiffness so rated thrust (~2.4 MN) gives a published-order
     mean surge offset, and pitch stiffness so it gives a published-order mean
     pitch angle. THIS is the physically decisive quantity for the study.
  2. **Natural periods** — set the resonant amplification of wave and control
     forcing. We fit the effective (rigid+added) inertias so the surge and pitch
     periods match published VolturnUS-S values.

Mass/inertia come from published properties. Every anchor/target is a named field
below and recorded in ASSUMPTIONS.md; nothing is hidden in the solver.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from physics.constants import G, RHO_WATER


@dataclass(frozen=True)
class Platform:
    # Geometry (NREL/TP-5000-76773) ------------------------------------------
    water_depth_m: float = 200.0
    draft_m: float = 20.0
    freeboard_m: float = 15.0
    column_diameter_m: float = 12.5
    column_spacing_m: float = 51.75      # centre column to offset column [m]
    n_offset_columns: int = 3
    displacement_m3: float = 20206.0     # displaced volume [m^3]
    fairlead_depth_m: float = 14.0       # fairlead below MSL [m] (mooring attach)

    # Mass properties (published; see ASSUMPTIONS.md for the small direct-drive
    # generator-inertia folding and COG derivation) --------------------------
    platform_ballast_mass_kg: float = 17.839e6   # hull steel + ballast
    tower_mass_kg: float = 1.263e6
    rna_mass_kg: float = 1.017e6                  # rotor-nacelle assembly
    system_cog_z_m: float = -1.0                  # system COG above MSL [m] (derived)

    # Reference site / hub for coupling --------------------------------------
    hub_height_m: float = 150.0                   # thrust application height [m]

    # ---- Fit anchors (representative, documented) ---------------------------
    rated_thrust_ref_N: float = 2.4e6             # anchor thrust for static offsets
    static_surge_at_rated_m: float = 20.0         # published-order mean surge offset
    static_pitch_at_rated_deg: float = 5.5        # published-order mean pitch lean
    surge_natural_period_s: float = 120.0         # target (VolturnUS-S literature)
    pitch_natural_period_s: float = 28.0          # target (VolturnUS-S literature)
    surge_damping_ratio: float = 0.05             # modal zeta (radiation+viscous+mooring)
    pitch_damping_ratio: float = 0.03             # modal zeta

    @property
    def total_mass_kg(self) -> float:
        return self.platform_ballast_mass_kg + self.tower_mass_kg + self.rna_mass_kg

    @property
    def waterplane_area_m2(self) -> float:
        # Four surface-piercing columns of equal diameter.
        n = self.n_offset_columns + 1
        return n * math.pi / 4.0 * self.column_diameter_m ** 2


@dataclass
class ReducedPlatformModel:
    """Fitted 2-DOF (surge, pitch) linear system: (M+A) q'' + B q' + C q = F.

    Ordering of the 2-vector q is [surge x (m), pitch theta (rad)]. All matrices
    are 2x2. ``fit_report`` records how each coefficient was obtained.
    """

    M: np.ndarray            # effective mass matrix (rigid + added) [kg ; kg m ; kg m^2]
    B: np.ndarray            # damping matrix
    C: np.ndarray            # restoring (mooring + hydrostatic) matrix
    z_hub_m: float           # thrust lever arm (hub height) [m]
    platform: Platform
    fit_report: dict = field(default_factory=dict)

    def natural_periods_s(self) -> np.ndarray:
        """Undamped natural periods from the generalized eigenproblem (C, M)."""
        w2, _ = _gen_eig(self.C, self.M)
        w2 = np.sort(np.clip(w2, 1e-12, None))
        return 2.0 * math.pi / np.sqrt(w2)

    def static_response(self, thrust_N: float) -> tuple[float, float]:
        """Static [surge_m, pitch_rad] under a steady horizontal thrust at the hub."""
        f = np.array([thrust_N, thrust_N * self.z_hub_m])
        q = np.linalg.solve(self.C, f)
        return float(q[0]), float(q[1])


def _gen_eig(C: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve C phi = w^2 M phi -> eigenvalues w^2 and eigenvectors."""
    from scipy.linalg import eig
    w2, phi = eig(C, M)
    return np.real(w2), np.real(phi)


def build_reduced_model(platform: Platform | None = None,
                        surge_period_s: float | None = None,
                        pitch_period_s: float | None = None) -> ReducedPlatformModel:
    """Build the fitted 2-DOF model. See module docstring for the fit logic."""
    p = platform or VOLTURNUS_S
    Ts = surge_period_s or p.surge_natural_period_s
    Tp = pitch_period_s or p.pitch_natural_period_s

    # --- Restoring stiffness from static-lean anchors (the decisive quantity) ---
    k_surge = p.rated_thrust_ref_N / p.static_surge_at_rated_m            # N/m
    pitch_rad = math.radians(p.static_pitch_at_rated_deg)
    c_pitch = p.rated_thrust_ref_N * p.hub_height_m / pitch_rad           # N m / rad
    # Mooring surge-pitch coupling: horizontal mooring load acts at fairlead depth.
    c_couple = -k_surge * p.fairlead_depth_m                             # N/rad (= N m/m)
    C = np.array([[k_surge, c_couple],
                  [c_couple, c_pitch]])

    # --- Effective inertias from natural-period targets ---
    ws, wp = 2 * math.pi / Ts, 2 * math.pi / Tp
    m_eff = k_surge / ws ** 2                     # surge effective mass (rigid+added)
    i_eff = c_pitch / wp ** 2                     # pitch effective inertia (rigid+added)
    # Small mass coupling from COG offset (product of inertia); minor but included.
    m_couple = p.total_mass_kg * p.system_cog_z_m
    M = np.array([[m_eff, m_couple],
                  [m_couple, i_eff]])

    # --- Damping via modal damping ratios (radiation + viscous + mooring) ---
    # Build B so each uncoupled mode carries its specified zeta: B = 2 zeta sqrt(k m).
    b_surge = 2.0 * p.surge_damping_ratio * math.sqrt(k_surge * m_eff)
    b_pitch = 2.0 * p.pitch_damping_ratio * math.sqrt(c_pitch * i_eff)
    B = np.array([[b_surge, 0.0],
                  [0.0, b_pitch]])

    model = ReducedPlatformModel(M=M, B=B, C=C, z_hub_m=p.hub_height_m, platform=p)
    # Verify & record the fit.
    per = model.natural_periods_s()
    xs, ts = model.static_response(p.rated_thrust_ref_N)
    model.fit_report = {
        "target_surge_period_s": Ts,
        "target_pitch_period_s": Tp,
        "fitted_natural_periods_s": sorted(per.tolist()),
        "k_surge_N_per_m": k_surge,
        "c_pitch_Nm_per_rad": c_pitch,
        "surge_effective_mass_kg": m_eff,
        "pitch_effective_inertia_kgm2": i_eff,
        "static_surge_at_rated_m": xs,
        "static_pitch_at_rated_deg": math.degrees(ts),
        "added_mass_fraction_surge": m_eff / p.total_mass_kg - 1.0,
    }
    return model


# Frozen reference singleton.
VOLTURNUS_S = Platform()
