"""Reduced-order platform motion — surge + pitch, wave + thrust forced (§5.5).

Equation of motion (2-DOF, q = [surge x (m), pitch theta (rad)]):
    (M) q'' + B q' + C q = F_thrust(t) [1, z_hub]^T + F_wave(t)
where (M, B, C) are the fitted matrices from ``models.volturnus_s`` (mass+added mass,
damping, restoring). Thrust acts at hub height, producing both surge and a large pitch
moment. The slow, control-driven drift that this thrust transient excites — riding under
the fast wave motion — is what does the novel hang-off fatigue, so it is preserved (never
high-passed away).

Wave excitation (reduced-order, documented — not a diffraction solution)
------------------------------------------------------------------------
Froude-Krylov / Morison-inertia forces on the four surface-piercing columns, with the
depth integral of the deep-water kinematics evaluated in closed form per wave component,
and per-column wave phase (k x_col) so the platform's fore-aft extent produces a genuine
wave pitch moment. Viscous drag is folded into the modal damping (B). This yields
wave-frequency surge and pitch of the right order; it is labelled reduced-order.

One-way coupling (platform -> cable) is used in the cable module: the cable's reaction on
a ~20 000 t platform is negligible. Stated in EXTENSIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.volturnus_s import Platform, ReducedPlatformModel
from physics.constants import G, RHO_WATER
from physics.waves import WaveField

# State layout within the platform sub-vector: [x, x_dot, theta, theta_dot].
IDX_X = 0
IDX_XDOT = 1
IDX_THETA = 2
IDX_THETADOT = 3
N_PLATFORM_STATES = 4

# Morison / Froude-Krylov coefficients (reduced-order excitation).
C_M = 2.0     # inertia coefficient (Ca = 1 for a circular cylinder)


def _column_x_positions(p: Platform) -> np.ndarray:
    """Fore-aft (x) positions of the columns [m]: 3 offset at 0/120/240 deg + centre."""
    r = p.column_spacing_m
    angles = np.array([0.0, 120.0, 240.0]) * np.pi / 180.0
    x_offsets = r * np.cos(angles)
    return np.concatenate([x_offsets, [0.0]])   # + centre column at x=0


@dataclass
class WaveForcing:
    """Precomputed wave surge force and pitch moment on a time grid (interpolated)."""
    t: np.ndarray
    F_surge: np.ndarray      # [N]
    M_pitch: np.ndarray      # [N m]

    def at(self, t: float) -> np.ndarray:
        fs = np.interp(t, self.t, self.F_surge)
        mp = np.interp(t, self.t, self.M_pitch)
        return np.array([fs, mp])


def build_wave_forcing(platform: Platform, wave: WaveField,
                       t_end: float, dt_force: float = 0.05) -> WaveForcing:
    """Precompute Froude-Krylov/Morison wave surge force and pitch moment vs time."""
    tg = np.arange(0.0, t_end + dt_force, dt_force)
    F_surge, M_pitch = wave_forces_at(platform, wave, tg)
    return WaveForcing(t=tg, F_surge=F_surge, M_pitch=M_pitch)


def wave_forces_at(platform: Platform, wave: WaveField, tg: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Froude-Krylov/Morison surge force [N] and pitch moment [N m] at arbitrary times."""
    x_cols = _column_x_positions(platform)
    D = platform.column_diameter_m
    A_col = np.pi / 4.0 * D ** 2
    draft = platform.draft_m
    w = wave.w
    k = wave.k
    amp = wave.amp
    ph = wave.phase

    # Closed-form depth integrals per component (deep-water decay e^{k z}, z in [-draft,0]).
    I0 = (1.0 - np.exp(-k * draft)) / k                       # ∫ e^{kz} dz
    I1 = (-1.0 / k ** 2) + np.exp(-k * draft) * (draft / k + 1.0 / k ** 2)  # ∫ z e^{kz} dz

    inertia_coeff = RHO_WATER * C_M * A_col                    # per column, per unit accel-integral

    F_surge = np.zeros_like(tg)
    M_pitch = np.zeros_like(tg)
    # Vectorize over components; loop over the (only 4) columns and time in chunks.
    WT = np.outer(tg, w)                                       # (T, N)
    for xc in x_cols:
        theta_arg = WT - k * xc + ph                           # (T, N)
        sin_t = np.sin(theta_arg)
        cos_t = np.cos(theta_arg)
        # Horizontal Morison-inertia force on this column: rho Cm A ∫ a_x dz.
        # a_x = amp w^2 e^{kz} sin(theta) -> depth integral uses I0.
        Fh = inertia_coeff * (sin_t * (amp * w ** 2 * I0)).sum(axis=1)
        F_surge += Fh
        # Overturning moment from the horizontal force (depth integral of z*a_x -> I1).
        Mh = inertia_coeff * (sin_t * (amp * w ** 2 * I1)).sum(axis=1)
        # Vertical Froude-Krylov (fluctuating buoyancy) at this column: rho g A_col eta.
        eta_col = (cos_t * amp).sum(axis=1)
        Fv = RHO_WATER * G * A_col * eta_col
        # Pitch moment about MSL origin: horizontal contribution + vertical*(-x_col).
        M_pitch += Mh - Fv * xc
    return F_surge, M_pitch


def platform_accel(model: ReducedPlatformModel, q: np.ndarray, qdot: np.ndarray,
                   F_thrust_N: float, F_wave: np.ndarray) -> np.ndarray:
    """Return [x_ddot, theta_ddot] from the 2-DOF EOM."""
    F_ext = np.array([F_thrust_N, F_thrust_N * model.z_hub_m]) + F_wave
    rhs = F_ext - model.B @ qdot - model.C @ q
    return np.linalg.solve(model.M, rhs)


def effective_pitch_damping(model: ReducedPlatformModel, daero_dxdot: float) -> float:
    """Effective pitch damping ratio including the aero feedback (negative-damping check).

    A rotor thrust that decreases with fore-aft velocity (dF/d(x_dot) < 0) *adds* damping;
    an aero/control feedback that increases thrust with velocity *removes* it and can drive
    the pitch mode unstable. ``daero_dxdot`` [N per (m/s)] is dF_thrust/d(x_dot_nacelle).

    The hub velocity in pitch is z_hub*theta_dot, so the aero force adds an equivalent
    pitch-damping term B55_aero = -daero_dxdot * z_hub^2 (thrust opposes hub motion when
    daero_dxdot<0 -> positive damping). Returns zeta_pitch of the pitch mode.
    """
    z = model.z_hub_m
    B55 = model.B[1, 1] - daero_dxdot * z ** 2
    I55 = model.M[1, 1]
    C55 = model.C[1, 1]
    zeta = B55 / (2.0 * np.sqrt(max(C55 * I55, 1e-9)))
    return float(zeta)
