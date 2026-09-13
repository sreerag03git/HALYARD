"""Lifelike, lit 3-D CAD assembly of the coupled system — all from real geometry (§8).

A markedly more detailed successor to :func:`app.viz3d.system_figure`: the same inputs and
behaviour, but built as a lit CAD assembly rather than a schematic. Everything is constructed
procedurally from the real reference objects (VolturnUS-S, IEA-15MW, the solved lazy-wave
cable) using the solid Mesh3d primitives in :mod:`app.meshes` — a semi-transparent sea
surface with waterline rings, a layered seabed, tapered/heave-plated columns and box
pontoons with gusset plates, a transition piece and bolted-flange tapered tower, a detailed
rotor-nacelle assembly, the colour-mapped dynamic cable with bend stiffener / buoyancy
modules / J-tube, three full catenary mooring lines to anchor blocks, and in-scene dimension
callouts. No illustrative or imported imagery — geometry only.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from app import meshes as M
from app.dwg_kit import (BUOY, CABLE_BLACK, SEABED, SEABED_LINE, STEEL, STEEL_D,
                         WATER)
from app.theme import ACCENT, GRID_LINE, INK
from models.dynamic_cable import REFERENCE_CABLE, DynamicCable
from models.iea15mw import IEA15MW, Turbine
from models.volturnus_s import Platform
from physics.cable import StaticShape, solve_static_shape

# Lit-material constants (match app.meshes so the whole scene shades consistently).
_LIGHT = dict(ambient=0.45, diffuse=0.85, specular=0.25, roughness=0.55, fresnel=0.15)
_LIGHTPOS = dict(x=200, y=400, z=600)

NACELLE = "#4A5764"      # nacelle housing
HUBCOL = "#3A4450"       # spinner / hub
BLADE = "#EDEFF1"        # blade shell (off-white composite)
MOORING = "#556070"      # mooring chain/wire tube
ANCHOR = "#42474D"       # anchor block


# ---------------------------------------------------------------------------
# Local geometry helpers (pure; no global state, no streamlit)
# ---------------------------------------------------------------------------
def _column_positions(p: Platform) -> list[tuple[float, float]]:
    """Offset-column centres on a circle of radius ``column_spacing`` + the centre column.

    Mirrors :func:`app.viz3d._column_positions` exactly (authoritative layout).
    """
    r = p.column_spacing_m
    ang = np.array([0.0, 120.0, 240.0]) * np.pi / 180.0
    xs = r * np.cos(ang)
    ys = r * np.sin(ang)
    return list(zip(xs, ys)) + [(0.0, 0.0)]


def _plane(z: float, xr, yr, color: str, opacity: float) -> go.Surface:
    """Flat quad at height ``z`` spanning (xr, yr) — sea surface / seabed sheet."""
    X, Y = np.meshgrid(np.linspace(*xr, 2), np.linspace(*yr, 2))
    Z = np.full_like(X, z)
    return go.Surface(x=X, y=Y, z=Z, showscale=False,
                      colorscale=[[0, color], [1, color]], opacity=opacity,
                      hoverinfo="skip")


def _ring_line(cx: float, cy: float, z: float, r: float, color: str,
               n: int = 40, width: int = 3) -> go.Scatter3d:
    """A horizontal circle (waterline mark) drawn on a column at height ``z``."""
    th = np.linspace(0, 2 * np.pi, n)
    return go.Scatter3d(x=cx + r * np.cos(th), y=cy + r * np.sin(th),
                        z=np.full(n, z), mode="lines",
                        line=dict(color=color, width=width), hoverinfo="skip",
                        showlegend=False)


def _oriented_cylinder(center, axis, r: float, half_len: float, color: str,
                       n: int = 20, opacity: float = 1.0, name: str = "") -> go.Mesh3d:
    """Solid capped cylinder centred at ``center`` with its axis along ``axis``.

    Complements :func:`app.meshes.solid_cylinder` (which is z-aligned only); used here for
    the along-shaft hub drum and the pitch-bearing rings that lie in the rotor plane.
    """
    center = np.asarray(center, float)
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, tmp); u /= np.linalg.norm(u) + 1e-9
    w = np.cross(axis, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.cos(th)[:, None] * u[None, :] + np.sin(th)[:, None] * w[None, :]
    c0 = center - axis * half_len
    c1 = center + axis * half_len
    bot = c0[None, :] + r * ring
    top = c1[None, :] + r * ring
    verts = np.vstack([bot, top, c0[None, :], c1[None, :]])
    icb, ict = 2 * n, 2 * n + 1
    faces = []
    for a in range(n):
        b = (a + 1) % n
        faces += [[a, b, n + b], [a, n + b, n + a]]     # side
        faces += [[icb, b, a]]                          # bottom cap
        faces += [[ict, n + a, n + b]]                  # top cap
    f = np.asarray(faces, int)
    v = verts
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                     color=color, opacity=opacity, flatshading=True, name=name,
                     lighting=_LIGHT, lightposition=_LIGHTPOS, hoverinfo="skip",
                     showlegend=False)


def _annotate(fig: go.Figure, texts: list[tuple], leaders: list[tuple]) -> None:
    """Add in-scene dimension text (Scatter3d) plus thin leader lines to the figure."""
    if texts:
        fig.add_trace(go.Scatter3d(
            x=[t[0] for t in texts], y=[t[1] for t in texts], z=[t[2] for t in texts],
            mode="text", text=[t[3] for t in texts],
            textfont=dict(size=10, color=INK), hoverinfo="skip", showlegend=False))
    for (p0, p1) in leaders:
        fig.add_trace(go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]], mode="lines",
            line=dict(color=INK, width=2, dash="dot"), hoverinfo="skip", showlegend=False))


# ---------------------------------------------------------------------------
# Rotor-nacelle assembly (reusable) — returns a list of go.Mesh3d
# ---------------------------------------------------------------------------
def nacelle_assembly(origin, hub_r: float = 3.2, nacelle_len: float = 18.0,
                     nacelle_w: float = 8.0, nacelle_h: float = 7.0,
                     rotor_radius: float = 120.0, root_chord: float = 5.4,
                     tip_chord: float = 0.6, precone_deg: float = 4.0,
                     azimuths: tuple = (90.0, 210.0, 330.0),
                     nacelle_color: str = NACELLE, hub_color: str = HUBCOL,
                     blade_color: str = BLADE, ring_color: str = STEEL_D) -> list:
    """Detailed rotor-nacelle assembly as a list of solid meshes (rotor faces +x).

    Nacelle housing box (aft of the hub), a downwind spinner nose cone, a short hub drum
    along the shaft, three pitch-bearing rings in the rotor plane, and the three tapered/
    twisted IEA-15MW blades (rotor Ø ``2*rotor_radius``). Returned as a list so
    :func:`system_figure_pro` can extend the figure with it and it can be reused elsewhere.
    """
    origin = np.asarray(origin, float)
    ox, oy, oz = origin
    parts: list = []

    # Nacelle housing: box extending aft (-x) from the hub plane.
    parts.append(M.box(ox - nacelle_len / 2.0, oy, oz, nacelle_len, nacelle_w, nacelle_h,
                       nacelle_color, name="nacelle"))
    # Hub drum along the shaft axis, then the downwind spinner nose cone (+x).
    parts.append(_oriented_cylinder(origin + np.array([hub_r * 0.5, 0, 0]),
                                    np.array([1.0, 0, 0]), hub_r, hub_r * 0.8, hub_color))
    parts.append(M.cone((ox + hub_r * 2.8, oy, oz), (ox + hub_r * 0.9, oy, oz),
                       hub_r, hub_color))
    # Three pitch-bearing rings in the rotor plane, one per blade root.
    for az in azimuths:
        a = np.radians(az)
        radial = np.array([0.0, np.cos(a), np.sin(a)])
        center = origin + radial * (hub_r * 1.15)
        parts.append(_oriented_cylinder(center, radial, hub_r * 0.85, 0.45, ring_color, n=16))
    # Three lofted, tapered, twisted blades.
    for az in azimuths:
        parts.append(M.blade_mesh(tuple(origin), az, rotor_radius - 3.0, blade_color,
                                  root_chord=root_chord, tip_chord=tip_chord,
                                  precone_deg=precone_deg, root_r=hub_r))
    return parts


# ---------------------------------------------------------------------------
# Full lit system model — supersedes app.viz3d.system_figure
# ---------------------------------------------------------------------------
def system_figure_pro(shape: StaticShape, cable: DynamicCable, platform: Platform,
                      stress_along: np.ndarray | None = None,
                      color_label: str = "curvature (1/m)",
                      turbine: Turbine | None = None) -> go.Figure:
    """Lifelike, lit CAD-style 3-D model of the coupled system from the real geometry.

    Same inputs/behaviour as :func:`app.viz3d.system_figure`, upgraded to a detailed lit
    assembly. ``stress_along`` (if given) colours the cable instead of curvature; ``turbine``
    defaults to the IEA-15MW reference.
    """
    turbine = turbine or IEA15MW
    depth = cable.water_depth_m
    extent_x = (-120.0, float(shape.x.max()) + 60.0)
    extent_y = (-100.0, 100.0)
    fig = go.Figure()

    # (1) Sea surface (semi-transparent) + layered seabed (soil slab + two sheets).
    fig.add_trace(_plane(0.0, extent_x, extent_y, WATER, 0.20))
    soil_cx = (extent_x[0] + extent_x[1]) / 2.0
    soil_cy = (extent_y[0] + extent_y[1]) / 2.0
    soil_sx = extent_x[1] - extent_x[0]
    soil_sy = extent_y[1] - extent_y[0]
    fig.add_trace(M.box(soil_cx, soil_cy, -depth - 2.0, soil_sx, soil_sy, 4.0, SEABED))
    fig.add_trace(_plane(-depth, extent_x, extent_y, SEABED, 0.85))
    fig.add_trace(_plane(-depth - 4.0, extent_x, extent_y, SEABED_LINE, 0.65))

    cols = _column_positions(platform)
    Rc = platform.column_diameter_m / 2.0
    draft, fb = platform.draft_m, platform.freeboard_m
    hub_h = platform.hub_height_m

    # (2) Pontoons (centre -> each offset column) with gusset plates at the joints.
    cx0, cy0 = cols[-1]
    for (cx, cy) in cols[:-1]:
        yaw = np.arctan2(cy, cx)
        length = np.hypot(cx, cy) - platform.column_diameter_m
        fig.add_trace(M.box((cx0 + cx) / 2.0, (cy0 + cy) / 2.0, -draft + 3.5,
                            length, platform.column_diameter_m * 0.9, 7.0, STEEL_D, yaw=yaw))
        ux, uy = cx / np.hypot(cx, cy), cy / np.hypot(cx, cy)
        for gx, gy in ((cx - ux * Rc, cy - uy * Rc), (cx0 + ux * Rc, cy0 + uy * Rc)):
            fig.add_trace(M.box(gx, gy, -draft + 7.5, 3.0, platform.column_diameter_m * 0.9,
                               2.0, STEEL, yaw=yaw))

    # (3) Columns as solid lit cylinders + wider heave-plate hint at each keel.
    for (cx, cy) in cols:
        fig.add_trace(M.solid_cylinder(cx, cy, -draft, fb, Rc, STEEL, name="column"))
        fig.add_trace(M.solid_cylinder(cx, cy, -draft, -draft + 1.6, Rc * 1.35, STEEL_D))
        fig.add_trace(_ring_line(cx, cy, 0.0, Rc * 1.01, WATER))   # waterline ring at z=0

    # (4) Transition piece + tapered tower (cone) with two bolted-flange rings.
    tp_top = fb + 6.0
    fig.add_trace(M.solid_cylinder(0, 0, fb, tp_top, 5.5, STEEL_D))          # transition piece
    fig.add_trace(M.cone((0, 0, hub_h), (0, 0, tp_top), 5.0, STEEL_D))       # tapered tower

    def _tower_r(z: float) -> float:
        return 5.0 * (hub_h - z) / (hub_h - tp_top)

    for zf in (tp_top + 0.5, tp_top + 0.42 * (hub_h - tp_top)):
        rf = _tower_r(zf)
        fig.add_trace(M.solid_cylinder(0, 0, zf - 0.35, zf + 0.35, rf + 0.45, STEEL))

    # (5) Rotor-nacelle assembly (upwind of the tower top).
    hub_origin = (4.0, 0.0, hub_h)
    for part in nacelle_assembly(hub_origin, rotor_radius=turbine.rotor_radius_m):
        fig.add_trace(part)

    # (6) Dynamic lazy-wave cable as a colour-mapped tube (curvature or supplied stress).
    inten = stress_along if stress_along is not None else shape.curvature
    cab_pts = np.column_stack([shape.x, np.zeros_like(shape.x), shape.z])
    fig.add_trace(M.tube(cab_pts, cable.outer_diameter_m * 6.0, intensity=np.asarray(inten),
                         colorscale="Turbo", n=10, colorbar_title=color_label, name="cable",
                         show=True))
    # Short I/J-tube guide sleeve wrapping the cable where it leaves the hang-off column.
    fig.add_trace(M.tube(cab_pts[:5], 0.55, color=STEEL_D, n=10))
    # Bend stiffener (tapered cone) at the hang-off.
    fig.add_trace(M.cone(tuple(cab_pts[3]), tuple(cab_pts[1]), 1.6, CABLE_BLACK))
    # Buoyancy modules: clamped short-cylinder pairs straddling the buoyancy section.
    L = cable.total_length_m
    b0, b1 = cable.buoyancy_start_frac * L, cable.buoyancy_end_frac * L
    bmask = (shape.s >= b0) & (shape.s <= b1)
    for i in np.where(bmask)[0][::3]:
        for dy in (-0.7, 0.7):
            fig.add_trace(M.solid_cylinder(shape.x[i], dy, shape.z[i] - 2.0,
                                           shape.z[i] + 2.0, 0.85, BUOY, n=12))

    # (7) Mooring: three FULL catenary tubes from the offset-column fairleads to anchor
    # blocks. Anchor radius is foreshortened to ~240 m for readability (real lines run
    # much farther); the line touches down onto the seabed before the anchor.
    fair_d = platform.fairlead_depth_m
    anchor_reach = 240.0
    s = np.linspace(0.0, 1.0, 44)
    td = 0.72                                   # touchdown fraction along the span
    for (cx, cy) in cols[:-1]:
        rr = np.hypot(cx, cy)
        ux, uy = cx / rr, cy / rr
        fx, fy = cx + ux * Rc, cy + uy * Rc     # fairlead on the outboard column face
        anx, any_ = fx + ux * anchor_reach, fy + uy * anchor_reach
        mx = fx + s * (anx - fx)
        my = fy + s * (any_ - fy)
        drop = np.clip(s / td, 0.0, 1.0) ** 1.8
        mz = -fair_d + (-depth + fair_d) * drop
        fig.add_trace(M.tube(np.column_stack([mx, my, mz]), 1.4, color=MOORING, n=7))
        fig.add_trace(M.box(anx, any_, -depth + 1.5, 5.0, 5.0, 3.0, ANCHOR))

    # (8) In-scene dimension callouts with leader lines.
    Rrot = turbine.rotor_radius_m
    xmid = float(shape.x.max()) * 0.5
    dim_x = extent_x[0] + 12.0
    texts = [
        (4.0, 0.0, hub_h + Rrot + 8.0, f"rotor Ø {2 * Rrot:.0f} m"),
        (dim_x - 6.0, 0.0, hub_h / 2.0, f"hub {hub_h:.0f} m"),
        (0.0, 0.0, 5.0, "MSL"),
        (xmid, -18.0, -depth / 2.0, f"water depth {depth:.0f} m"),
        (cols[0][0] + Rc + 10.0, cols[0][1], -draft / 2.0, f"draft {draft:.0f} m"),
        (cols[1][0], cols[1][1] + Rc + 8.0, fb + 3.0, f"column Ø {platform.column_diameter_m:.1f} m"),
        (cols[0][0] / 2.0, -24.0, -draft, f"col. spacing {platform.column_spacing_m:.2f} m"),
        (cab_pts[1][0] + 8.0, 0.0, cab_pts[1][2] + 2.0, "hang-off"),
    ]
    leaders = [
        ((dim_x, 0.0, 0.0), (dim_x, 0.0, hub_h)),                       # hub-height stick
        ((4.0, 0.0, hub_h + Rrot), (4.0, 0.0, hub_h - Rrot)),           # rotor diameter
        ((xmid, -18.0, 0.0), (xmid, -18.0, -depth)),                    # water-depth stick
        ((cols[0][0] + Rc, cols[0][1], -draft), (cols[0][0] + Rc, cols[0][1], 0.0)),  # draft
        ((0.0, -24.0, -draft), (cols[0][0], -24.0, -draft)),            # column-spacing span
        ((cab_pts[1][0], 0.0, cab_pts[1][2]), (cab_pts[1][0] + 8.0, 0.0, cab_pts[1][2] + 2.0)),
    ]
    _annotate(fig, texts, leaders)

    _scene(fig, extent_x, extent_y, depth)
    return fig


def _scene(fig: go.Figure, extent_x, extent_y, depth: float) -> None:
    """Apply the true-scale 3-D scene layout (data aspect, house camera, transparent paper)."""
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID_LINE,
                       showspikes=False),
            yaxis=dict(title="y (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID_LINE,
                       showspikes=False),
            zaxis=dict(title="z (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID_LINE,
                       showspikes=False),
            aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=0.55),
                                           up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=10, b=0), height=640, paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(255,255,255,0.75)", bordercolor=GRID_LINE, borderwidth=1),
    )
