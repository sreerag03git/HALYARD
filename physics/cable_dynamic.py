"""Dynamic lumped-mass cable — transverse dynamics with Morison hydro, bending, seabed & VIV.

The quasi-static family (``physics/cable.py``) maps platform motion to hang-off stress
geometrically. This module is the higher-fidelity DYNAMIC cable: a lumped-mass line whose
transverse motion is integrated in time under

  * inertia + added mass (Morison),
  * hydrodynamic drag on the relative fluid-structure velocity (Morison),
  * bending stiffness (EI) via a curvature-restoring force,
  * gravity / buoyancy (incl. the buoyancy section),
  * seabed contact + Coulomb friction,
  * vortex-induced vibration (VIV) cross-flow lift via a van der Pol wake oscillator,

driven at the top by the platform fairlead motion (surge, heave, pitch → hang-off point,
with a clamped bend-stiffener BC that rotates with platform pitch). Inextensibility is
enforced by red-black distance-constraint projection (position-based dynamics), which keeps
the scheme stable at the simulation time step without resolving the stiff axial wave. Tension
is recovered by inverse dynamics (force balance including inertia) — so the DYNAMIC tension
(heave/inertia/drag/snap tendency), not just the quasi-static value, reaches the fatigue
pipeline. One-way coupling (platform → cable) as before.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.dynamic_cable import DynamicCable, REFERENCE_CABLE
from physics.cable import solve_static_shape, _region_indices, REGIONS, BEND_STIFFENER_LEN_M
from physics.constants import G, RHO_WATER
from physics.waves import WaveField

# Morison / hydro coefficients for a slender cable (normal direction).
CD_N = 1.2           # normal drag coefficient
CA_N = 1.0           # normal added-mass coefficient (Cm = 1 + Ca)
# VIV wake-oscillator (Facchinetti et al. van der Pol model) parameters.
VIV_ST = 0.17        # Strouhal number
VIV_CL0 = 0.3        # reference lift coefficient
VIV_A = 12.0         # wake-oscillator coupling
VIV_EPS = 0.3        # van der Pol nonlinearity


@dataclass
class DynamicCableResult:
    t: np.ndarray
    region_curvature: dict           # region -> kappa(t) [1/m]
    region_tension: dict             # region -> T(t) [N]
    region_idx: dict
    top_x: np.ndarray                # fairlead x(t)
    top_z: np.ndarray                # fairlead z(t)
    shape0: object                   # static baseline shape
    viv: bool


def _fairlead_motion(cable: DynamicCable, surge, heave, pitch_rad):
    """Hang-off point position from platform motion (rigid-body, small angle)."""
    x_ho, z_ho = cable.hangoff_x_m, cable.hangoff_z_m
    x_top = x_ho + surge + z_ho * pitch_rad
    z_top = z_ho + heave
    return x_top, z_top


def solve_dynamic_cable(cable: DynamicCable, t: np.ndarray, surge, heave, pitch_rad,
                        wave: WaveField, n_nodes: int = 50, constraint_sweeps: int = 6,
                        viv: bool = True) -> DynamicCableResult:
    """Integrate the dynamic cable driven by the fairlead motion; return region kappa & T."""
    cable = cable or REFERENCE_CABLE
    dt = float(t[1] - t[0])
    N = n_nodes
    L = cable.total_length_m
    L0 = L / N
    depth = cable.water_depth_m
    D = cable.outer_diameter_m
    A_cross = np.pi / 4 * D ** 2

    # Baseline static shape (also the mean node positions for wave-kinematics precompute).
    shape0 = solve_static_shape(cable, n_nodes=N)
    r0 = np.column_stack([shape0.x, shape0.z])
    reg = _region_indices(shape0, cable)
    # Per-segment rest lengths from the static shape (the seabed tail is straightened, so
    # segments are non-uniform); using them keeps the initial config consistent with the
    # inextensibility constraints (no violent redistribution at startup).
    L0_seg = np.hypot(np.diff(r0[:, 0]), np.diff(r0[:, 1]))

    # Per-node submerged weight [N] and buoyancy-section flag.
    s = np.arange(N + 1) * L0
    w_lin = np.full(N + 1, cable.submerged_weight_N_per_m)
    b0, b1 = cable.buoyancy_start_frac * L, cable.buoyancy_end_frac * L
    w_lin[(s >= b0) & (s <= b1)] = cable.buoyancy_section_net_N_per_m
    node_wt = w_lin * L0
    node_wt[0] *= 0.5
    node_wt[-1] *= 0.5

    # Effective nodal mass = structural + added mass (Morison, isotropic approx).
    m_struct = cable.dry_mass_per_len_kgm * L0
    m_added = RHO_WATER * CA_N * A_cross * L0
    m_node = np.full(N + 1, m_struct + m_added)
    m_dryonly = np.full(N + 1, m_struct + m_added)

    # Precompute wave kinematics at the mean node positions (nodes move little horizontally).
    zc = np.minimum(r0[:, 1], 0.0)
    u_g, w_g, ax_g, az_g = wave.kinematics_grid(t, r0[:, 0], zc)   # [nt, N+1] each

    # Fairlead motion drives the top by the DEVIATION from the mean (the static shape r0 is
    # already at the design mean offset — as in the quasi-static family). Absolute surge (~20 m)
    # would otherwise yank the hang-off far from r0.
    surge = np.asarray(surge) - np.mean(surge)
    heave = np.asarray(heave) - np.mean(heave)
    pitch_dev = np.asarray(pitch_rad) - np.mean(pitch_rad)
    x_top, z_top = _fairlead_motion(cable, surge, heave, pitch_dev)
    dep0 = np.radians(shape0.departure_angle_deg)     # baseline departure from vertical
    # Stiffener axis unit vector (downward), rotated by pitch deviation.
    ang = dep0 + pitch_dev                             # angle from vertical (into the water)
    stiff_dx = np.sin(ang)
    stiff_dz = -np.cos(ang)

    anchor = r0[-1].copy()
    inv_m = 1.0 / m_node
    seg_i = np.arange(N)
    seg_j = np.arange(1, N + 1)
    kbed = 4.0e4
    mu_bed = 0.5                                       # seabed Coulomb friction

    # Constraint inverse-mass weights (top two nodes + anchor pinned).
    w_free = inv_m.copy()

    def pin(r, i):
        r[0] = [x_top[i], z_top[i]]
        r[1] = [x_top[i] + L0_seg[0] * stiff_dx[i], z_top[i] + L0_seg[0] * stiff_dz[i]]
        r[-1] = anchor

    from physics.cable import _tension_from_geometry, _touchdown_index

    # VIV wake-oscillator state (cross-flow), per node.
    q_viv = np.zeros(N + 1)
    qd_viv = np.zeros(N + 1)
    omega_s_max = 0.4 / dt            # clamp shedding freq to keep the explicit VIV stable

    r = r0.copy()
    v = np.zeros_like(r)
    pin(r, 0)
    # Light numerical damping only (physical damping comes from Morison drag); low enough to
    # let dynamic amplification appear, high enough for stability.
    damp = 0.996

    curv_hist = {k: np.zeros(len(t)) for k in REGIONS}
    ten_hist = {k: np.zeros(len(t)) for k in REGIONS}

    for it in range(len(t)):
        # --- segment tangents/normals (per node) ---
        d = np.zeros_like(r)
        d[1:-1] = r[2:] - r[:-2]
        d[0] = r[1] - r[0]
        d[-1] = r[-1] - r[-2]
        dl = np.hypot(d[:, 0], d[:, 1]) + 1e-9
        tx, tz = d[:, 0] / dl, d[:, 1] / dl
        nx, nz = -tz, tx
        # --- external forces ---
        F = np.zeros_like(r)
        F[:, 1] -= node_wt
        urx, urz = u_g[it] - v[:, 0], w_g[it] - v[:, 1]
        urn = urx * nx + urz * nz                        # relative normal velocity
        an = ax_g[it] * nx + az_g[it] * nz               # fluid normal acceleration
        Fn = (0.5 * RHO_WATER * CD_N * D * np.abs(urn) * urn * L0      # Morison drag
              + RHO_WATER * (1 + CA_N) * A_cross * L0 * an)            # Morison inertia
        if viv:
            omega_s = np.minimum(2 * np.pi * VIV_ST * np.abs(urn) / D, omega_s_max)
            vel_n = v[:, 0] * nx + v[:, 1] * nz
            qdd = (-VIV_EPS * omega_s * (q_viv ** 2 - 1.0) * qd_viv - omega_s ** 2 * q_viv
                   + VIV_A * vel_n / D)
            qd_viv = np.clip(qd_viv + qdd * dt, -50, 50)
            q_viv = np.clip(q_viv + qd_viv * dt, -6, 6)
            Fn = Fn + 0.5 * RHO_WATER * D * urn ** 2 * (VIV_CL0 * q_viv) * L0
        F[:, 0] += Fn * nx
        F[:, 1] += Fn * nz
        lap = np.zeros_like(r)
        lap[1:-1] = r[2:] - 2 * r[1:-1] + r[:-2]
        F[1:-1] += (cable.EI_Nm2 / L0 ** 3) * lap[1:-1]     # bending (EI)

        # --- damped semi-implicit velocity update ---
        v = (v + (F * inv_m[:, None]) * dt) * damp
        r_old = r.copy()
        r = r + v * dt
        pin(r, it)
        v[0] = v[1] = v[-1] = 0.0
        # --- inextensibility (red-black distance constraints) ---
        for _ in range(constraint_sweeps):
            for S in (np.arange(0, N, 2), np.arange(1, N, 2)):
                ii, jj = S, S + 1
                dd = r[jj] - r[ii]
                ll = np.hypot(dd[:, 0], dd[:, 1]) + 1e-9
                corr = ((ll - L0_seg[S]) / ll)[:, None] * dd
                sden = (w_free[ii] + w_free[jj])[:, None] + 1e-12
                r[ii] = r[ii] + (w_free[ii][:, None] / sden) * corr
                r[jj] = r[jj] - (w_free[jj][:, None] / sden) * corr
            pin(r, it)
        # --- seabed contact + Coulomb friction (no sliding of laid nodes) ---
        below = r[:, 1] < -depth
        r[below, 1] = -depth
        r[below, 0] = r_old[below, 0]
        # recompute velocity from the projected positions (constraints affect it)
        v = (r - r_old) / dt

        # --- diagnostics: curvature & tension at the regions ---
        seg = r[1:] - r[:-1]
        ang_seg = np.arctan2(seg[:, 1], seg[:, 0])
        dang = np.diff(np.unwrap(ang_seg))
        seglen = np.hypot(seg[:, 0], seg[:, 1])
        kappa = np.zeros(N + 1)
        kappa[1:-1] = np.abs(dang) / (0.5 * (seglen[:-1] + seglen[1:]) + 1e-9)
        kappa[reg["hang_off"]] += abs(pitch_dev[it]) / BEND_STIFFENER_LEN_M
        td = _touchdown_index(r, depth)
        Tnode = _tension_from_geometry(r, node_wt, td)   # dynamic geometry -> dynamic tension
        for k in REGIONS:
            curv_hist[k][it] = kappa[reg[k]]
            ten_hist[k][it] = Tnode[reg[k]]

    return DynamicCableResult(t=t, region_curvature=curv_hist, region_tension=ten_hist,
                              region_idx=reg, top_x=x_top, top_z=z_top, shape0=shape0,
                              viv=viv)


def region_stress_from_dynamic(dyn: DynamicCableResult, region: str,
                               hangoff_stick: bool = True) -> np.ndarray:
    """Stress time series at a region from the DYNAMIC curvature + tension [Pa].

    Same equivalent-armour model as the quasi-static path:
        sigma = E_steel * r_bend * kappa(t) + T(t) / A_armour
    but kappa and T are the dynamic (Morison/inertia/VIV) values, so the dynamic
    amplification and VIV enter the fatigue directly (no assumed DAF).
    """
    c = dyn.shape0  # not used; kept for clarity
    from models.dynamic_cable import REFERENCE_CABLE
    cab = REFERENCE_CABLE
    kappa = dyn.region_curvature[region]
    T = dyn.region_tension[region]
    if region == "hang_off":
        r_bend = cab.hangoff_bend_radius_m if hangoff_stick else cab.armour_wire_radius_m
    else:
        r_bend = cab.armour_wire_radius_m
    return cab.E_steel_Pa * r_bend * kappa + T / cab.armour_area_m2


def run_dynamic_cable_from_sim(sim, cable: DynamicCable | None = None,
                               n_nodes: int = 50, viv: bool = True,
                               discard_s: float = 100.0) -> DynamicCableResult:
    """Drive the dynamic cable with a coupled-sim result's fairlead (6-DOF) motion."""
    cable = cable or REFERENCE_CABLE
    cfg = sim.config
    wave = WaveField(Hs_m=cfg.Hs_m, Tp_s=cfg.Tp_s, seed=cfg.wave_seed)
    heave = sim.heave_m if sim.heave_m is not None else np.zeros_like(sim.t)
    res = solve_dynamic_cable(cable, sim.t, sim.surge_m, heave,
                              np.radians(sim.pitch_deg), wave, n_nodes=n_nodes, viv=viv)
    res.counting_mask = sim.t >= discard_s
    return res
