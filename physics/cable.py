"""Dynamic cable — lazy-wave static configuration, curvature and stress (§5.6).

Static shape
------------
The lazy-wave profile is solved as a lumped-mass line by vectorized dynamic relaxation:
N nodes, tension-only axial springs, per-node net submerged weight (negative in the
buoyancy section so it arches into a hog/sag), and a seabed contact. The cable is
near-inextensible (T/EA ~ 1e-4), so a reduced solve-stiffness recovers the same geometry
robustly; the physical top tension is then computed from force balance on the converged
shape (EA-independent).

Dynamic response — quasi-static family + DAF (the stated live method, §5.6 option a)
------------------------------------------------------------------------------------
Platform motion moves the hang-off (top) node. Hang-off fatigue is driven by the CHANGE
in the cable's departure angle over the bend-stiffener length (the static catenary
curvature at a steep hang-off is low; the bending happens in the stiffener as the angle
swings). We therefore re-solve the static shape across a range of top-node horizontal
offsets to build the quasi-static family alpha(x_top), T_top(x_top), and the sag/hog/
touchdown curvatures, then drive it with the platform-motion time series and apply a
dynamic amplification factor. Because the buoyancy section decouples the lower cable, the
hang-off angle is far more motion-sensitive than sag/hog/touchdown — the concentration is
itself a finding (and makes the lay a mitigation variable).

Stress (simplified equivalent-armour model, see models.dynamic_cable):
    sigma(t) = E_steel * R_armour * kappa(t) + T(t) / A_armour
One-way coupling (platform -> cable) only; the cable's reaction on the platform is
negligible (EXTENSIONS.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.dynamic_cable import DynamicCable, REFERENCE_CABLE

# Critical regions along the cable.
REGIONS = ("hang_off", "sag", "hog", "touchdown")

# Bend-stiffener length over which a hang-off departure-angle change is distributed [m].
BEND_STIFFENER_LEN_M = 4.0


@dataclass
class StaticShape:
    x: np.ndarray            # node x [m]
    z: np.ndarray            # node z [m] (0 = MSL)
    s: np.ndarray            # arc length from hang-off [m]
    tension_N: np.ndarray    # tension at each node [N]
    curvature: np.ndarray    # geometric curvature at each node [1/m]
    departure_angle_deg: float   # angle from vertical at the hang-off [deg]
    top_tension_N: float
    touchdown_idx: int
    converged: bool
    residual: float


def solve_static_shape(cable: DynamicCable, x_top: float | None = None,
                       z_top: float | None = None, anchor_x: float | None = None,
                       anchor_z: float | None = None, n_nodes: int = 80,
                       max_iter: int = 8000, constraint_sweeps: int = 20,
                       smooth_w: float = 0.06,
                       r_init: np.ndarray | None = None) -> StaticShape:
    """Solve the lazy-wave static shape by Position-Based Dynamics (inextensible cable).

    Verlet integration under gravity/buoyancy + seabed contact, with iterative distance-
    constraint projection (Jacobi sweeps with inverse-mass weighting) enforcing segment
    inextensibility exactly. This is the standard, robust way to relax a rope/cable to
    its hanging equilibrium; it is stable and yields a smooth quasi-static family. Tension
    is recovered from force balance on the converged shape.
    """
    x_top = cable.hangoff_x_m if x_top is None else x_top
    z_top = cable.hangoff_z_m if z_top is None else z_top
    L = cable.total_length_m
    depth = cable.water_depth_m
    N = n_nodes
    L0 = L / N
    s = np.arange(N + 1) * L0

    w_lin = np.full(N + 1, cable.submerged_weight_N_per_m)
    b0, b1 = cable.buoyancy_start_frac * L, cable.buoyancy_end_frac * L
    in_buoy = (s >= b0) & (s <= b1)
    w_lin[in_buoy] = cable.buoyancy_section_net_N_per_m
    node_wt = w_lin * L0
    node_wt[0] *= 0.5
    node_wt[-1] *= 0.5

    # The subsea anchor is FIXED on the seabed (absolute position). When the platform
    # hang-off moves (x_top changes), the horizontal span changes — that is what alters the
    # hang-off geometry and drives the fatigue.
    if anchor_x is None:
        anchor_x = cable.hangoff_x_m + cable.horizontal_layout_m
    anchor = np.array([anchor_x, -depth if anchor_z is None else anchor_z])
    top = np.array([x_top, z_top])

    # Inverse mass: fixed ends immovable (w=0).
    w = np.ones(N + 1)
    w[0] = 0.0
    w[-1] = 0.0
    seg_i = np.arange(N)         # segment j connects node j and j+1
    seg_j = np.arange(1, N + 1)
    denom = w[seg_i] + w[seg_j] + 1e-12

    if r_init is not None and r_init.shape == (N + 1, 2):
        r = r_init.copy()
    else:
        frac = (s / L)[:, None]
        r = top[None, :] + frac * (anchor - top)[None, :]
    r[0] = top
    r[-1] = anchor
    r_prev = r.copy()

    # Net vertical load direction (buoyancy nodes push up); normalized so the per-step
    # gravitational move is a small fraction of L0 (a controlled descent rate — magnitude
    # is arbitrary in PBD, only the balance of loads and the length constraint matter).
    accel = np.zeros((N + 1, 2))
    accel[:, 1] = -node_wt
    dt = 0.15
    damp = 0.9
    accel *= (0.08 * L0) / (np.max(np.abs(accel)) * dt ** 2 + 1e-12)
    residual = np.inf
    converged = False
    for it in range(max_iter):
        # Verlet step for free nodes.
        nxt = r + (r - r_prev) * damp + accel * dt ** 2
        nxt[0] = top
        nxt[-1] = anchor
        r_prev = r
        r = nxt
        # Light bending resistance (EI > 0): Laplacian smoothing of free interior nodes,
        # which suppresses the spurious zig-zag buckling of the slack seabed-laid portion
        # and gently rounds the touchdown, as bending stiffness physically does. Applied
        # BEFORE the length constraint so the constraint projection has the last word on
        # segment length (conserves total length).
        if smooth_w > 0:
            lap = 0.5 * (r[:-2] + r[2:]) - r[1:-1]
            r[1:-1] = r[1:-1] + smooth_w * lap
            r[0] = top
            r[-1] = anchor
        # Distance-constraint projection (Jacobi sweeps) — enforces inextensibility.
        for _ in range(constraint_sweeps):
            d = r[seg_j] - r[seg_i]
            length = np.sqrt((d ** 2).sum(axis=1)) + 1e-12
            corr = ((length - L0) / length)[:, None] * d
            delta = np.zeros_like(r)
            np.add.at(delta, seg_i, (w[seg_i] / denom)[:, None] * corr)
            np.add.at(delta, seg_j, -(w[seg_j] / denom)[:, None] * corr)
            r = r + delta
            r[0] = top
            r[-1] = anchor
        # Seabed contact (vertical clamp only; the bending smoothing keeps the laid tail
        # straight without a monotonic-x hack that would force-stretch the slack).
        np.maximum(r[:, 1], -depth, out=r[:, 1])
        r[-1] = anchor
        if it % 50 == 0:
            move = float(np.max(np.abs(r - r_prev)))
            if it > 100 and move < 1e-5:
                converged = True
                residual = move
                break
            residual = move

    # Length-restoration polish: forward-backward Gauss-Seidel distance projection
    # (Jakobsen's rope method). Using updated positions within a sweep propagates the
    # correction along the chain in O(N) rather than Jacobi's O(N^2), so it converges the
    # inextensibility tightly in a few hundred sweeps (early stop on the worst segment).
    r = _gauss_seidel_length(r, L0, w, top, anchor, depth, max_sweeps=1500, tol=5e-4)

    # Straighten the seabed-laid tail: physically the slack cable lies straight on the flat
    # seabed from the touchdown to the anchor. The discrete solver otherwise folds the excess
    # back on itself (a spurious kink); replacing the tail with a straight lay removes it and
    # gives the correct zero curvature there. The suspended part (which carries the fatigue)
    # is untouched.
    b_end = int(np.searchsorted(s, cable.buoyancy_end_frac * L))
    near_bed = np.where((np.arange(N + 1) > b_end) & (r[:, 1] < -depth + 2.0))[0]
    if len(near_bed):
        td_bed = int(near_bed[0])
        if N - td_bed >= 1:
            r[td_bed:] = np.linspace(r[td_bed], anchor, N - td_bed + 1)
            r[td_bed:, 1] = -depth
            r[td_bed, 1] = r[td_bed, 1]   # touchdown node sits on the seabed

    touchdown_idx = _touchdown_index(r, depth)
    tension = _tension_from_geometry(r, node_wt, touchdown_idx)
    curvature = _curvature_from_shape(r, s)
    # Beyond touchdown the cable lies flat on the seabed (curvature ~ 0); zero any
    # discretization residual there so the MBR/curvature reflect the suspended cable.
    curvature[touchdown_idx + 1:] = 0.0
    dep_angle = _departure_angle_deg(r)
    return StaticShape(x=r[:, 0], z=r[:, 1], s=s, tension_N=tension,
                       curvature=curvature, departure_angle_deg=dep_angle,
                       top_tension_N=float(tension[0]), touchdown_idx=touchdown_idx,
                       converged=converged, residual=residual)


def _gauss_seidel_length(r: np.ndarray, L0: float, w: np.ndarray, top, anchor,
                         depth: float, max_sweeps: int = 400, tol: float = 1e-3) -> np.ndarray:
    """Red-black (even/odd) vectorized distance-constraint projection to L0 (fixed ends).

    Even and odd segments share no nodes, so each colour updates in parallel; alternating
    them gives Gauss-Seidel-rate convergence with fully vectorized NumPy (fast), replacing a
    slow per-node Python loop.
    """
    N = len(r) - 1
    even = np.arange(0, N, 2)
    odd = np.arange(1, N, 2)
    for _ in range(max_sweeps):
        for S in (even, odd):
            i, j = S, S + 1
            d = r[j] - r[i]
            l = np.sqrt((d ** 2).sum(axis=1)) + 1e-12
            corr = ((l - L0) / l)[:, None] * d
            sden = (w[i] + w[j])[:, None] + 1e-12
            r[i] = r[i] + (w[i][:, None] / sden) * corr
            r[j] = r[j] - (w[j][:, None] / sden) * corr
        r[0] = top
        r[-1] = anchor
        np.maximum(r[:, 1], -depth, out=r[:, 1])
        seg = np.sqrt((np.diff(r, axis=0) ** 2).sum(axis=1))
        if np.max(np.abs(seg - L0)) / L0 < tol:
            break
    return r


def _tension_from_geometry(r: np.ndarray, node_wt: np.ndarray, td_idx: int) -> np.ndarray:
    """Cable tension [N] from force balance: T(s) = sqrt(H^2 + V(s)^2).

    Horizontal tension H is constant along a cable with no horizontal applied load;
    vertical tension V(s) is the net suspended weight carried above arc-length s. H is
    fixed by the top-segment geometry (H = V_top / tan(angle from horizontal)).
    """
    N = len(r) - 1
    # Vertical load carried above each node = cumulative net weight from touchdown up.
    V = np.zeros(N + 1)
    acc = 0.0
    for i in range(td_idx, -1, -1):
        acc += node_wt[i]
        V[i] = acc
    V_top = max(V[0], 1.0)
    d = r[1] - r[0]
    ang_horiz = np.arctan2(abs(d[1]), abs(d[0]) + 1e-9)   # top segment angle from horizontal
    H = V_top / max(np.tan(ang_horiz), 1e-3)
    T = np.sqrt(H ** 2 + V ** 2)
    T[td_idx + 1:] = H       # on the seabed the tension is ~horizontal
    return T


def _curvature_from_shape(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Discrete curvature [1/m] = turning angle between segments / mean segment length."""
    d = np.diff(r, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    dang = np.diff(np.unwrap(ang))
    seg_len = np.sqrt((d ** 2).sum(axis=1))
    mean_len = 0.5 * (seg_len[:-1] + seg_len[1:]) + 1e-9
    kappa_internal = np.abs(dang) / mean_len
    kappa = np.zeros(len(r))
    kappa[1:-1] = kappa_internal
    return kappa


def _departure_angle_deg(r: np.ndarray) -> float:
    """Angle of the first cable segment from the vertical at the hang-off [deg]."""
    d = r[1] - r[0]
    ang_from_vertical = np.degrees(np.arctan2(abs(d[0]), abs(d[1]) + 1e-9))
    return float(ang_from_vertical)


def _touchdown_index(r: np.ndarray, depth: float) -> int:
    on_bed = np.where(r[:, 1] <= -depth + 0.5)[0]
    return int(on_bed[0]) if len(on_bed) else len(r) - 1


def _region_indices(shape: StaticShape, cable: DynamicCable) -> dict:
    """Locate the four critical regions along the lazy-wave.

    hog  = crest (local z-max) within/near the buoyancy section;
    sag  = trough (local z-min) of the lower catenary between the hog and touchdown;
    touchdown = last suspended node; hang_off = first cable node below the fixed top.
    """
    td = shape.touchdown_idx
    z = shape.z
    L = cable.total_length_m
    b0, b1 = cable.buoyancy_start_frac * L, cable.buoyancy_end_frac * L
    buoy = np.where((shape.s >= b0 - 8) & (shape.s <= b1 + 8))[0]
    hog_idx = int(buoy[np.argmax(z[buoy])]) if len(buoy) else max(td // 2, 1)
    # Sag bend = trough of the upper catenary between the hang-off and the hog.
    upper = np.arange(2, max(hog_idx, 3))
    sag_idx = int(upper[np.argmin(z[upper])]) if len(upper) else max(hog_idx // 2, 1)
    return {"hang_off": 1, "sag": sag_idx, "hog": hog_idx, "touchdown": max(td - 1, 1)}


@dataclass
class QuasiStaticFamily:
    """Hang-off departure angle, region tensions and curvatures vs the top-node offset.

    Built by re-solving the static shape across a range of hang-off horizontal offsets
    ``dx`` about the design mean (warm-started for speed). Applied to a platform-motion
    time series, this maps the slow control drift + fast wave motion onto hang-off
    curvature/tension, so the mean-stress shift (§5.7) emerges naturally.
    """
    dx: np.ndarray                       # top-node horizontal offset from design mean [m]
    angle_deg: np.ndarray                # hang-off departure angle from vertical [deg]
    region_T: dict                       # region -> tension array [N] vs dx
    region_kappa: dict                   # region -> geometric curvature array [1/m] vs dx
    base_shape: StaticShape              # shape at dx = 0
    cable: DynamicCable
    shapes_xz: np.ndarray = None         # (n_dx, n_nodes+1, 2) full node positions vs dx

    def angle_at(self, dx: np.ndarray) -> np.ndarray:
        return np.interp(dx, self.dx, self.angle_deg)

    def shape_at(self, dx_val: float) -> np.ndarray:
        """Full cable node positions at a top offset, linearly interpolated (for animation)."""
        j = np.clip(np.searchsorted(self.dx, dx_val) - 1, 0, len(self.dx) - 2)
        d0, d1 = self.dx[j], self.dx[j + 1]
        t = 0.0 if d1 == d0 else np.clip((dx_val - d0) / (d1 - d0), 0.0, 1.0)
        return (1 - t) * self.shapes_xz[j] + t * self.shapes_xz[j + 1]


def build_quasistatic_family(cable: DynamicCable | None = None,
                             dx_max_m: float = 3.0, n: int = 13,
                             n_nodes: int = 80) -> QuasiStaticFamily:
    """Re-solve the lazy-wave across top offsets dx in [-dx_max, +dx_max] (warm-started)."""
    cable = cable or REFERENCE_CABLE
    dxs = np.linspace(-dx_max_m, dx_max_m, n)
    base = solve_static_shape(cable, n_nodes=n_nodes)
    reg = _region_indices(base, cable)
    angle = np.zeros(n)
    T = {k: np.zeros(n) for k in REGIONS}
    K = {k: np.zeros(n) for k in REGIONS}
    # Sweep outward from dx=0 in both directions, warm-starting from the neighbour.
    order = sorted(range(n), key=lambda i: abs(dxs[i] - 0.0))
    r_warm = {0.0: np.column_stack([base.x, base.z])}
    shapes = {}
    for i in order:
        dx = dxs[i]
        # nearest already-solved dx as warm start
        done = list(r_warm.keys())
        nearest = min(done, key=lambda d: abs(d - dx))
        sh = solve_static_shape(cable, x_top=cable.hangoff_x_m + dx, n_nodes=n_nodes,
                                r_init=r_warm[nearest])
        r_warm[dx] = np.column_stack([sh.x, sh.z])
        shapes[i] = sh
    for i in range(n):
        sh = shapes[i]
        r = _region_indices(sh, cable)
        angle[i] = sh.departure_angle_deg
        for name in REGIONS:
            T[name][i] = sh.tension_N[r[name]]
            K[name][i] = sh.curvature[r[name]]
    # sort by dx for interp
    idx = np.argsort(dxs)
    shapes_xz = np.array([np.column_stack([shapes[i].x, shapes[i].z]) for i in range(n)])[idx]
    return QuasiStaticFamily(dx=dxs[idx], angle_deg=angle[idx],
                             region_T={k: T[k][idx] for k in REGIONS},
                             region_kappa={k: K[k][idx] for k in REGIONS},
                             base_shape=base, cable=cable, shapes_xz=shapes_xz)


def region_stress_timeseries(family: QuasiStaticFamily, region: str,
                             surge_t: np.ndarray, pitch_rad_t: np.ndarray,
                             daf: float = 1.2, hangoff_stick: bool = True
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stress, curvature and tension time series at a region from platform surge + pitch.

    The hang-off horizontal offset dx = (surge - mean) + z_attach*(pitch - mean) drives the
    quasi-static family. At the HANG-OFF the dominant control-sensitive driver is the
    platform PITCH rotating the bend stiffener (rigidly attached to the platform) relative
    to the cable's far-field direction: the stiffener curvature change is
    [ Delta(alpha_natural) - Delta(pitch) ] / L_stiffener. Other regions use their geometric
    curvature from the family (translation only; the buoyancy section decouples them from
    platform rotation). ``daf`` amplifies the fluctuating part (§5.6 option a).

    Bending lever by region: the HANG-OFF carries the highest tension, so its armour tends
    to STICK (no slip) — the physical reason hang-offs are fatigue-critical — and uses the
    armour pitch radius (``hangoff_stick=True``, default). The lower-tension sag/hog/
    touchdown SLIP and use the wire radius. Set ``hangoff_stick=False`` for the hang-off
    slip lower bound (sensitivity study).
    """
    c = family.cable
    surge = np.asarray(surge_t, float)
    pitch = np.asarray(pitch_rad_t, float)
    z_att = c.hangoff_z_m
    dx = (surge - surge.mean()) + z_att * (pitch - pitch.mean())
    T = np.interp(dx, family.dx, family.region_T[region])
    kappa_geom = np.interp(dx, family.dx, family.region_kappa[region])
    if region == "hang_off":
        alpha0 = family.angle_at(np.array([0.0]))[0]
        d_alpha = np.radians(family.angle_at(dx) - alpha0)      # natural-angle change [rad]
        d_pitch = pitch - pitch.mean()                          # stiffener rotation [rad]
        kappa_stiff = (d_alpha - d_pitch) / BEND_STIFFENER_LEN_M
        kappa = kappa_geom + kappa_stiff
        # Default: calibrated effective (partial-slip) lever. hangoff_stick=False switches to
        # the full-slip wire-radius lower bound (sensitivity study).
        r_bend = c.hangoff_bend_radius_m if hangoff_stick else c.armour_wire_radius_m
    else:
        kappa = kappa_geom
        r_bend = c.armour_wire_radius_m
    # DAF on the fluctuating (de-meaned) part of curvature and tension.
    kappa = kappa.mean() + daf * (kappa - kappa.mean())
    T = T.mean() + daf * (T - T.mean())
    sigma = c.E_steel_Pa * r_bend * kappa + T / c.armour_area_m2
    return sigma, kappa, T

