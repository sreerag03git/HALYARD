"""Dimensioned general-arrangement drawings (§8) — blueprint style, true scale.

Two CAD-quality general-arrangement sheets built entirely from the real
VolturnUS-S / IEA-15MW / lazy-wave-cable numbers and the solved static shape:

  * ``ga_elevation`` — side elevation (x span, z elevation; MSL z=0) with the
    hull columns, tower, rotor, and solved dynamic-cable profile, fully
    dimensioned. Supersedes ``app.drawings.general_arrangement``.
  * ``plan_view``    — top-down general arrangement: hull footprint, mooring
    spread and dynamic-cable route, with radial / angular / diameter dimensions.

All drafting furniture (frame, title block, dimensions, hatching, scale bar,
north arrow, leaders) comes from ``app.dwg_kit``. No streamlit import; every
public function returns a ``go.Figure`` and is independently callable.
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from app import dwg_kit as K
from app.dwg_kit import (BP_INK, BP_LINE, BUOY, CABLE_BLACK, SEABED, SEABED_LINE,
                         STEEL, STEEL_D, WATER)
from models.dynamic_cable import REFERENCE_CABLE, DynamicCable
from models.iea15mw import IEA15MW, Turbine
from models.volturnus_s import VOLTURNUS_S, Platform
from physics.cable import StaticShape, _region_indices, solve_static_shape

NACELLE = "#4A5764"
_ROTOR_LINE = "#33414E"


# --- small local drafting helpers (pure; operate on the passed figure) -----
def _circle(fig: go.Figure, cx: float, cy: float, r: float, fill: str,
            line: str = BP_LINE, width: float = 1.2, layer: str = "above") -> None:
    """Filled circle from a bounding box (used for columns in plan)."""
    fig.add_shape(type="circle", x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                  line=dict(color=line, width=width), fillcolor=fill, layer=layer)


def _rect(fig: go.Figure, x0: float, y0: float, x1: float, y1: float, fill: str,
          line: str = BP_LINE, width: float = 1.0) -> None:
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                  line=dict(color=line, width=width), fillcolor=fill)


def _poly(fig: go.Figure, pts, fill: str, line: str = BP_LINE, width: float = 1.0) -> None:
    """Closed filled polygon from a list of (x, y) vertices."""
    path = "M " + " L ".join(f"{px},{py}" for px, py in pts) + " Z"
    fig.add_shape(type="path", path=path, line=dict(color=line, width=width), fillcolor=fill)


def _beam(fig: go.Figure, x0: float, y0: float, x1: float, y1: float, width: float,
          fill: str, line: str = BP_LINE) -> None:
    """Rectangular beam of given width running from (x0,y0) to (x1,y1) (plan pontoon)."""
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L * width / 2.0, dx / L * width / 2.0
    _poly(fig, [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny),
                (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)], fill, line)


def _anchor(fig: go.Figure, x: float, y: float, tag: str) -> None:
    """Plate-anchor symbol, apex toward the platform, with a tag label."""
    L = math.hypot(x, y) or 1.0
    ux, uy = -x / L, -y / L                       # inward (toward origin)
    nx, ny = -uy, ux                              # normal
    s = 11.0
    apex = (x + ux * s, y + uy * s)
    b0 = (x - ux * s * 0.4 + nx * s * 0.7, y - uy * s * 0.4 + ny * s * 0.7)
    b1 = (x - ux * s * 0.4 - nx * s * 0.7, y - uy * s * 0.4 - ny * s * 0.7)
    _poly(fig, [apex, b0, b1], BP_INK, line=BP_INK, width=0.8)
    fig.add_annotation(x=x - ux * 20, y=y - uy * 20, text=tag, showarrow=False,
                       font=dict(size=8, color=BP_INK))


def _break_mark(fig: go.Figure, x: float, y: float, ax: float, ay: float) -> None:
    """Double-slash break symbol across a truncated line at (x,y) (direction ax,ay)."""
    L = math.hypot(ax, ay) or 1.0
    ux, uy = ax / L, ay / L
    nx, ny = -uy, ux
    d = 6.0
    for off in (-3.0, 3.0):
        cx, cy = x + ux * off, y + uy * off
        fig.add_shape(type="line", x0=cx - nx * d - ux * 2, y0=cy - ny * d - uy * 2,
                      x1=cx + nx * d + ux * 2, y1=cy + ny * d + uy * 2,
                      line=dict(color=BP_INK, width=1.0))


# ---------------------------------------------------------------------------
# Sheet 1 — side elevation
# ---------------------------------------------------------------------------
def ga_elevation(shape: StaticShape, cable: DynamicCable, platform: Platform,
                 turbine: Turbine | None = None) -> go.Figure:
    """Dimensioned side-elevation GA drawing (blueprint style, true scale).

    x is the horizontal span, z the elevation with MSL at z=0. Draws the seabed,
    hull columns/pontoon, tapered tower with bolted flanges, nacelle + rotor, and
    the solved lazy-wave cable (buoyancy section highlighted, four regions
    leadered), with a full dimension set, title block, notes and scale bar.
    """
    turbine = turbine or IEA15MW
    depth = platform.water_depth_m
    fb, draft = platform.freeboard_m, platform.draft_m
    hub = turbine.hub_height_m
    R = turbine.rotor_radius_m
    cd = platform.column_diameter_m
    sp = platform.column_spacing_m
    reg = _region_indices(shape, cable)

    xmax = float(shape.x.max())
    xr = (-185.0, xmax + 135.0)
    yr = (-depth - 58.0, hub + R + 60.0)
    dx = xr[1] - xr[0]

    fig = go.Figure()
    K.blueprint_axes(fig, xr, yr, height=660, equal=True)

    # --- water column + seabed --------------------------------------------
    wx0, wx1 = xr[0] + 0.05 * dx, xr[1] - 0.05 * dx
    fig.add_shape(type="rect", x0=wx0, y0=-depth, x1=wx1, y1=0.0, line=dict(width=0),
                  fillcolor="rgba(174,198,222,0.16)", layer="below")
    K.hatch_band(fig, wx0, wx1, -depth - 12.0, -depth, angle_deg=45.0, spacing=9.0,
                 color=SEABED_LINE, fill=SEABED)
    fig.add_shape(type="line", x0=wx0, y0=-depth, x1=wx1, y1=-depth,
                  line=dict(color=SEABED_LINE, width=1.2), layer="below")
    K.waterline(fig, wx0, wx1, z=0.0, label="MSL")

    # --- hull columns (true x-projections) + keel pontoon band -------------
    # Offset columns lie on a circle of radius sp at 0/120/240 deg; their x
    # projections are +sp, -sp/2, -sp/2. Draw the extreme front (+sp), the rear
    # pair (-sp/2, coincident) and the centre column (hang-off).
    x_front = sp
    x_rear = sp * math.cos(math.radians(120.0))       # = -sp/2
    col_x = (x_rear, 0.0, x_front)
    _rect(fig, min(col_x) - cd / 2 - 1.0, -draft, max(col_x) + cd / 2 + 1.0,
          -draft + 6.0, STEEL_D)                       # submerged pontoon band (keel)
    for cx in col_x:
        _rect(fig, cx - cd / 2, -draft, cx + cd / 2, fb, STEEL)
    K.centerline(fig, 0.0, -draft - 4.0, 0.0, hub + R + 8.0)   # tower/rotor centreline

    # --- transition piece + tapered tower + bolted flanges ----------------
    _rect(fig, -6.0, fb, 6.0, fb + 6.0, STEEL_D)               # transition piece
    tp_top = fb + 6.0
    wb, wt = 5.0, 2.6                                          # tower half-widths
    _poly(fig, [(-wb, tp_top), (wb, tp_top), (wt, hub), (-wt, hub)], STEEL)
    for zf in (55.0, 95.0, 135.0):
        frac = (zf - tp_top) / (hub - tp_top)
        w = wb + (wt - wb) * frac
        _rect(fig, -w - 0.5, zf - 1.1, w + 0.5, zf + 1.1, STEEL_D, width=0.8)

    # --- nacelle + spinner + rotor circle + three blades ------------------
    _rect(fig, -6.0, hub - 4.0, 12.0, hub + 4.0, NACELLE)      # nacelle envelope
    _poly(fig, [(12.0, hub - 3.2), (20.0, hub), (12.0, hub + 3.2)], "#5A6672")  # spinner
    th = np.linspace(0.0, 2.0 * np.pi, 120)
    fig.add_trace(go.Scatter(x=R * np.cos(th), y=hub + R * np.sin(th), mode="lines",
                             line=dict(color=_ROTOR_LINE, width=1.3), hoverinfo="skip",
                             showlegend=False))
    for a in (90.0, 210.0, 330.0):
        fig.add_shape(type="line", x0=0.0, y0=hub, x1=(R - 4.0) * math.cos(math.radians(a)),
                      y1=hub + (R - 4.0) * math.sin(math.radians(a)),
                      line=dict(color=_ROTOR_LINE, width=3.0))

    # --- solved dynamic-cable profile + buoyancy section + regions ---------
    L = cable.total_length_m
    b0, b1 = cable.buoyancy_start_frac * L, cable.buoyancy_end_frac * L
    bmask = (shape.s >= b0) & (shape.s <= b1)
    fig.add_trace(go.Scatter(x=shape.x, y=shape.z, mode="lines",
                             line=dict(color=CABLE_BLACK, width=2.6), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=shape.x[bmask], y=shape.z[bmask], mode="lines",
                             line=dict(color=BUOY, width=6.5), hoverinfo="skip",
                             showlegend=False))
    _leaders = {"hang_off": (-46, 40), "sag": (-30, -46), "hog": (0, 52),
                "touchdown": (44, -40)}
    for name, i in reg.items():
        ldx, ldy = _leaders.get(name, (36, 36))
        K.leader(fig, float(shape.x[i]), float(shape.z[i]),
                 name.replace("_", "-").upper(), dx=ldx, dy=ldy, size=9)

    # --- dimension set (real values) --------------------------------------
    K.dim_v(fig, 0.0, hub, x=-105.0, text=f"HUB {hub:.0f} m")
    K.dim_v(fig, -depth, 0.0, x=xr[0] + 34.0, text=f"WATER DEPTH {depth:.0f} m")
    K.dim_h(fig, -R, R, y=hub + R + 26.0, text=f"ROTOR DIA {2 * R:.0f} m", ext=26.0)
    K.dim_v(fig, 0.0, fb, x=x_front + 26.0, text=f"FB {fb:.0f} m", ext=26.0 - cd / 2)
    K.dim_v(fig, -draft, 0.0, x=x_front + 48.0, text=f"DRAFT {draft:.0f} m",
            ext=48.0 - cd / 2)
    K.dim_v(fig, cable.hangoff_z_m, 0.0, x=22.0, text=f"HANG-OFF {abs(cable.hangoff_z_m):.0f} m")
    K.dim_h(fig, -cd / 2, cd / 2, y=-draft - 15.0, text=f"COL DIA {cd:.1f} m")
    td_x = float(shape.x[reg["touchdown"]])
    K.dim_h(fig, 0.0, td_x, y=-depth - 22.0,
            text=f"HANG-OFF → TOUCHDOWN {td_x:.0f} m")

    # --- title block, revisions, notes, scale bar -------------------------
    K.frame(fig, xr, yr, zones=True)
    K.title_block(fig, xr, yr, title="GENERAL ARRANGEMENT — ELEVATION",
                  subtitle="IEA-15MW / UMaine VolturnUS-S / 66 kV lazy-wave dynamic cable",
                  dwg_no="HAL-GA-001", scale="1:NTS", rev="A",
                  extra="geometry computed (solved static shape)")
    K.revision_table(fig, xr, yr, rows=[("A", "issued from computed geometry")])
    K.notes_block(fig, xr, yr, notes=[
        "Dimensions in metres; elevations relative to MSL (z = 0).",
        "Cable profile is the solved lazy-wave static shape; regions computed.",
        "Rotor, tower and hull to VolturnUS-S / IEA-15-240-RWT reference data.",
        "Not to scale on the sheet — drawn true-scale, aspect preserved.",
    ], corner="top-left")
    K.scale_bar(fig, x=xr[0] + 40.0, y=-depth - 42.0, length_m=100.0, n=4, unit="m")
    return fig


# ---------------------------------------------------------------------------
# Sheet 2 — plan view
# ---------------------------------------------------------------------------
def plan_view(cable: DynamicCable, platform: Platform,
              turbine: Turbine | None = None) -> go.Figure:
    """Top-down general-arrangement plan: hull footprint, mooring spread, cable route.

    Draws the authoritative column layout (centre column + three offset columns
    on a pitch circle of radius ``column_spacing_m`` at 0/120/240 deg, joined by
    three pontoons), the 120-deg mooring spread (foreshortened to a break, real
    radius disclosed), fairlead delta plates, and the dynamic-cable route out to
    ``horizontal_layout_m``, with diameter / radial / angular dimensions.
    """
    turbine = turbine or IEA15MW
    R = platform.column_spacing_m
    Rc = platform.column_diameter_m / 2.0
    lay = cable.horizontal_layout_m
    angs = (0.0, 120.0, 240.0)
    offsets = [(R * math.cos(math.radians(a)), R * math.sin(math.radians(a))) for a in angs]

    moor_draw = 230.0                                 # drawn (foreshortened) radius
    xr = (-260.0, 265.0)
    yr = (-262.0, 262.0)
    dx = xr[1] - xr[0]

    fig = go.Figure()
    K.blueprint_axes(fig, xr, yr, height=640, equal=True)

    # --- column pitch circle (dash-dot reference) -------------------------
    th = np.linspace(0.0, 2.0 * np.pi, 160)
    fig.add_trace(go.Scatter(x=R * np.cos(th), y=R * np.sin(th), mode="lines",
                             line=dict(color=BP_LINE, width=0.7, dash="dashdot"),
                             hoverinfo="skip", showlegend=False))

    # --- pontoons (centre -> each offset), then columns on top ------------
    for (ox, oy) in offsets:
        _beam(fig, 0.0, 0.0, ox, oy, width=9.0, fill=STEEL_D, line=BP_LINE)
    for (ox, oy) in offsets:
        _circle(fig, ox, oy, Rc, STEEL)
    _circle(fig, 0.0, 0.0, Rc, STEEL)

    # --- mooring spread: fairlead delta plate + line + break + anchor -----
    for k, (a, (ox, oy)) in enumerate(zip(angs, offsets), start=1):
        ux, uy = math.cos(math.radians(a)), math.sin(math.radians(a))
        fx, fy = ox + ux * (Rc + 2.0), oy + uy * (Rc + 2.0)   # fairlead (outer edge)
        nx, ny = -uy, ux
        _poly(fig, [(fx + ux * 8.0, fy + uy * 8.0), (fx + nx * 4.5, fy + ny * 4.5),
                    (fx - nx * 4.5, fy - ny * 4.5)], STEEL_D)   # delta plate
        ax_, ay_ = ux * moor_draw, uy * moor_draw               # drawn anchor point
        fig.add_shape(type="line", x0=fx, y0=fy, x1=ax_, y1=ay_,
                      line=dict(color=STEEL, width=1.6))
        bxr = 0.72 * moor_draw
        _break_mark(fig, bxr * ux, bxr * uy, ux, uy)
        _anchor(fig, ax_, ay_, f"MA{k}")

    # --- dynamic-cable route (plan) out to the landing point --------------
    ca = 180.0                                        # route between MA2 and MA3
    cux, cuy = math.cos(math.radians(ca)), math.sin(math.radians(ca))
    lx, ly = lay * cux, lay * cuy
    fig.add_shape(type="line", x0=0.0, y0=0.0, x1=lx, y1=ly,
                  line=dict(color=CABLE_BLACK, width=2.6))
    r0, r1 = cable.buoyancy_start_frac * lay, cable.buoyancy_end_frac * lay
    fig.add_shape(type="line", x0=r0 * cux, y0=r0 * cuy, x1=r1 * cux, y1=r1 * cuy,
                  line=dict(color=BUOY, width=6.0))
    _circle(fig, lx, ly, 4.0, "#FFFFFF", line=CABLE_BLACK, width=1.4)
    K.leader(fig, lx, ly, "SUBSEA LANDING", dx=-6.0, dy=-34.0, size=9)
    K.leader(fig, 0.0, 0.0, "HANG-OFF (centre col.)", dx=28.0, dy=34.0, size=9)

    # --- dimensions -------------------------------------------------------
    K.dim_radial(fig, 0.0, 0.0, Rc, ang_deg=45.0, text=f"DIA {platform.column_diameter_m:.1f} m",
                 diameter=True)
    K.dim_radial(fig, 0.0, 0.0, R, ang_deg=60.0, text=f"{R:.2f} m")
    K.dim_angular(fig, 0.0, 0.0, R * 0.62, a0_deg=0.0, a1_deg=120.0, text="120°")
    K.dim_angular(fig, 0.0, 0.0, R * 0.62, a0_deg=120.0, a1_deg=240.0, text="120°")
    K.dim_radial(fig, 0.0, 0.0, moor_draw, ang_deg=0.0,
                 text="≈800 m ANCHOR (repr.)")
    K.dim_h(fig, 0.0, lx, y=ly - 26.0, text=f"CABLE ROUTE {lay:.0f} m", ext=26.0)

    # --- north arrow, frame, title block, notes, scale bar ----------------
    K.north_arrow(fig, x=xr[1] - 0.10 * dx, y=yr[1] - 0.20 * (yr[1] - yr[0]), size=34.0)
    K.frame(fig, xr, yr, zones=True)
    K.title_block(fig, xr, yr, title="GENERAL ARRANGEMENT — PLAN",
                  subtitle="UMaine VolturnUS-S / 3-line mooring / dynamic-cable route",
                  dwg_no="HAL-GA-002", scale="1:NTS", rev="A",
                  extra="geometry computed (published layout)")
    K.revision_table(fig, xr, yr, rows=[("A", "issued from computed geometry")])
    K.notes_block(fig, xr, yr, notes=[
        "Plan; dimensions in metres. Columns on a pitch circle R 51.75 m.",
        "Three pontoons join the centre column to the offset columns.",
        "Mooring lines shown foreshortened to a break — real radius ≈800 m (repr.).",
        "Dynamic cable routed to the landing at horizontal layout 165 m.",
    ], corner="top-left")
    K.scale_bar(fig, x=xr[0] + 40.0, y=yr[0] + 48.0, length_m=100.0, n=4, unit="m")
    return fig
