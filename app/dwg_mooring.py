"""Station-keeping mooring drawings — catenary line make-up + restoring law + plan (§8).

Three engineering figures built from the frozen VolturnUS-S reference numbers:

  * ``mooring_profile`` — dimensioned side elevation of ONE catenary mooring line
    from the fairlead (14 m below MSL on an offset column) down an analytic
    catenary to touchdown and along the seabed to a drag-embedment anchor, with a
    representative segment make-up (top chain -> polyester -> ground chain)
    overlaid as a schedule of lengths / diameters / MBL. Blueprint furniture from
    ``app.dwg_kit``; the horizontal is compressed (documented) so the ~800 m reach
    fits beside the 200 m column of water.
  * ``restoring_curve`` — the horizontal restoring characteristic vs surge offset,
    from the documented linear surge stiffness k = rated_thrust / mean-offset, with
    a representative hardening overlay for context.
  * ``mooring_plan`` — top-down station-keeping layout: the 3-line spread at 120 deg,
    fairlead delta plates, anchors (foreshortened, bearing + radius annotated), the
    mean-offset and watch circles, a drawn mooring schedule table, and angular /
    radial dimensions.

Segment diameters, lengths and MBL are representative station-keeping values
(not a NREL-published mooring schedule) and are flagged as such on the drawing.
No streamlit import; every public function returns a go.Figure.
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from app import dwg_kit as dk
from app import theme
from app.theme import ACCENT, GOOD, GREY, INK, WARN
from models.volturnus_s import VOLTURNUS_S, Platform

# Representative single-line make-up (documented as representative in the notes).
_CAT_PARAM_M = 300.0           # catenary parameter a = H/w [m] (representative)
_ANCHOR_RADIUS_M = 800.0       # nominal anchor radius from the column [m]
_PRETENSION_MN = 2.4           # representative fairlead pretension [MN]
_TOP_CHAIN = dict(name="studless top chain", dia_mm=185, mbl_mn=22.0, tag="R4 studless")
_POLYESTER = dict(name="polyester rope", dia_mm=200, mbl_mn=20.0, tag="buoyant midsection")
_GROUND_CHAIN = dict(name="ground / bottom chain", dia_mm=185, mbl_mn=22.0, tag="R4 studless")


def _catenary(platform: Platform, n: int = 240) -> tuple[np.ndarray, np.ndarray, float]:
    """Analytic suspended catenary from fairlead to touchdown; returns (x, z, x_td)."""
    depth = platform.water_depth_m
    fd = platform.fairlead_depth_m
    a = _CAT_PARAM_M
    h = depth - fd                              # fairlead height above seabed [m]
    x_td = a * math.acosh(1.0 + h / a)          # horizontal fairlead->touchdown [m]
    x = np.linspace(0.0, x_td, n)
    z = -depth + a * (np.cosh((x_td - x) / a) - 1.0)
    return x, z, x_td


def mooring_profile(platform: Platform = VOLTURNUS_S) -> go.Figure:
    """Catenary mooring-line make-up (elevation) with the segment schedule overlaid."""
    depth = platform.water_depth_m
    fd = platform.fairlead_depth_m
    cd = platform.column_diameter_m
    fb = platform.freeboard_m

    xs, zs, x_td = _catenary(platform)          # suspended catenary
    anchor_x = _ANCHOR_RADIUS_M

    xr = (-90.0, anchor_x + 80.0)
    yr = (-depth - 34.0, fb + 34.0)
    fig = go.Figure()
    dk.blueprint_axes(fig, xr, yr, height=620, equal=False)   # horizontal compressed

    # --- water + seabed ---------------------------------------------------
    dk.waterline(fig, xr[0] + 12, xr[1] - 12, z=0.0, label="MSL")
    dk.hatch_band(fig, xr[0] + 12, xr[1] - 12, -depth - 26, -depth,
                  angle_deg=55, color=dk.SEABED_LINE, fill=dk.SEABED)
    fig.add_shape(type="line", x0=xr[0] + 12, y0=-depth, x1=xr[1] - 12, y1=-depth,
                  line=dict(color=dk.SEABED_LINE, width=1.2))

    # --- offset column stub carrying the fairlead -------------------------
    fig.add_shape(type="rect", x0=-cd / 2, y0=-platform.draft_m, x1=cd / 2, y1=fb,
                  line=dict(color=dk.BP_LINE, width=1.2), fillcolor=dk.STEEL)
    dk.centerline(fig, 0, fb + 20, 0, -platform.draft_m - 6)
    fig.add_annotation(x=0, y=fb, text="offset column", showarrow=False, yshift=10,
                       font=dict(size=8, color=dk.BP_LINE))
    dk.dim_h(fig, -cd / 2, cd / 2, fb + 8, f"⌀ {cd:.1f} m column", ext=6.0)
    # Fairlead node.
    fig.add_trace(go.Scatter(x=[0.0], y=[-fd], mode="markers",
                             marker=dict(color=INK, size=9, symbol="square"),
                             hoverinfo="skip", showlegend=False))

    # --- mooring line drawn in three make-up segments ---------------------
    xg = np.linspace(x_td, anchor_x, 60)        # grounded run on the seabed
    zg = np.full_like(xg, -depth)
    x_all = np.concatenate([xs, xg])
    z_all = np.concatenate([zs, zg])
    # segment split points along the suspended catenary (representative make-up)
    x_c1 = 0.32 * x_td                          # top chain -> polyester
    seg_defs = (
        (x_all <= x_c1, _TOP_CHAIN, dk.STEEL_D, 3.4),
        ((x_all > x_c1) & (x_all <= x_td), _POLYESTER, ACCENT, 3.4),
        (x_all > x_td, _GROUND_CHAIN, dk.CABLE_BLACK, 3.4),
    )
    for mask, spec, color, w in seg_defs:
        fig.add_trace(go.Scatter(x=x_all[mask], y=z_all[mask], mode="lines",
                                 line=dict(color=color, width=w), hoverinfo="skip",
                                 showlegend=False))
    # Buoyancy-module beads along the polyester midsection (buoyant tag).
    poly_mask = (x_all > x_c1) & (x_all <= x_td)
    xb, zb = x_all[poly_mask], z_all[poly_mask]
    for k in np.linspace(0.12, 0.88, 5):
        i = int(k * (len(xb) - 1))
        fig.add_shape(type="circle", x0=xb[i] - 9, x1=xb[i] + 9,
                      y0=zb[i] - 5, y1=zb[i] + 5,
                      line=dict(color="#7A5A1E", width=0.8), fillcolor=dk.BUOY)

    # --- segment make-up callouts (leaders) -------------------------------
    def _seg_label(spec: dict) -> str:
        return (f"{spec['name']}<br>⌀ {spec['dia_mm']} mm · MBL {spec['mbl_mn']:.0f} MN"
                f"<br>({spec['tag']})")

    i_top = int(0.16 * len(xs))
    dk.leader(fig, xs[i_top], zs[i_top], _seg_label(_TOP_CHAIN),
              dx=140, dy=48, color=dk.STEEL_D, size=9)
    i_poly = int(0.62 * len(xs))
    dk.leader(fig, xs[i_poly], zs[i_poly], _seg_label(_POLYESTER),
              dx=150, dy=52, color=ACCENT, size=9)
    i_gnd = int(0.55 * len(xg))
    dk.leader(fig, xg[i_gnd], zg[i_gnd], _seg_label(_GROUND_CHAIN),
              dx=-30, dy=44, color=dk.CABLE_BLACK, size=9)

    # --- touchdown + anchor -----------------------------------------------
    fig.add_trace(go.Scatter(x=[x_td], y=[-depth], mode="markers",
                             marker=dict(color=WARN, size=9), hoverinfo="skip",
                             showlegend=False))
    dk.leader(fig, x_td, -depth, "touchdown point (TDP)", dx=10, dy=-40,
              color=WARN, size=9)
    _draw_anchor(fig, anchor_x, -depth)
    dk.leader(fig, anchor_x, -depth + 4, "drag-embedment anchor", dx=-40, dy=34,
              color=INK, size=9)

    # --- dimensions --------------------------------------------------------
    dk.dim_v(fig, 0.0, -fd, xr[0] + 46, f"{fd:.0f} m", ext=44.0)
    dk.dim_v(fig, 0.0, -depth, xr[0] + 20, f"WD {depth:.0f} m", ext=18.0)
    dk.dim_h(fig, 0.0, x_td, -depth - 14, f"suspended reach {x_td:.0f} m", ext=-14.0)
    dk.dim_h(fig, x_td, anchor_x, -depth - 14, f"ground reach {anchor_x - x_td:.0f} m",
             ext=-14.0)
    dk.dim_h(fig, 0.0, anchor_x, fb + 20, f"nominal anchor radius {anchor_x:.0f} m",
             ext=14.0)

    # Pretension callout at the fairlead.
    dk.leader(fig, 0.0, -fd, f"fairlead pretension ≈ {_PRETENSION_MN:.1f} MN (repr.)",
              dx=120, dy=-30, color=INK, size=9)

    # --- drawing furniture -------------------------------------------------
    dk.frame(fig, xr, yr)
    dk.notes_block(fig, xr, yr, [
        "One of three mooring lines shown (120° spread); geometry mirrored about the column.",
        "Segment make-up (chain / polyester / MBL) is REPRESENTATIVE, not a published schedule.",
        "Suspended profile is an analytic catenary, parameter a = H/w = "
        f"{_CAT_PARAM_M:.0f} m (repr.).",
        "Horizontal axis COMPRESSED to fit the reach; vertical is true elevation.",
        "Fairlead depth and water depth are frozen VolturnUS-S values.",
    ], corner="top-left")
    dk.revision_table(fig, xr, yr, [("A", "issued — representative make-up")])
    dk.title_block(fig, xr, yr, title="STATION-KEEPING MOORING — LINE MAKE-UP",
                   subtitle="VolturnUS-S semi-sub · catenary chain-polyester-chain",
                   dwg_no="HAL-MOR-001", scale="NTS (x compressed)", rev="A",
                   extra="computed catenary geometry")
    return fig


def _draw_anchor(fig: go.Figure, x: float, z: float) -> None:
    """Procedural drag-embedment anchor (shank + fluke) embedded at the seabed."""
    # shank rising slightly from the seabed toward the line
    fig.add_shape(type="line", x0=x, y0=z, x1=x - 16, y1=z + 6,
                  line=dict(color=INK, width=2.2))
    # embedded fluke (filled triangle below the seabed)
    path = (f"M {x - 22},{z - 2} L {x + 10},{z - 14} "
            f"L {x + 4},{z - 22} L {x - 30},{z - 8} Z")
    fig.add_shape(type="path", path=path, line=dict(color=INK, width=1.0),
                  fillcolor=dk.STEEL_D)


def restoring_curve(platform: Platform = VOLTURNUS_S) -> go.Figure:
    """Horizontal mooring+hydrostatic restoring force vs surge offset."""
    k_surge = platform.rated_thrust_ref_N / platform.static_surge_at_rated_m   # N/m
    x0 = platform.static_surge_at_rated_m
    f0 = platform.rated_thrust_ref_N

    x = np.linspace(0.0, 2.0 * x0, 200)
    f_lin = k_surge * x / 1e6                                   # linear restoring [MN]
    c_hard = 9.0e-5                                             # repr. hardening coeff
    f_nl = k_surge * (x + c_hard * x ** 3) / 1e6               # representative hardening

    fig = go.Figure()
    # Representative hardening (context) — drawn faint, behind the linear law.
    fig.add_trace(go.Scatter(x=x, y=f_nl, mode="lines", name="hardening (representative)",
                             line=dict(color=GREY, width=1.6, dash="dot")))
    # Documented linear surge stiffness.
    fig.add_trace(go.Scatter(x=x, y=f_lin, mode="lines",
                             name=f"linear k = {k_surge/1e3:.0f} kN/m",
                             line=dict(color=ACCENT, width=3)))
    # Mean operating point at rated thrust.
    fig.add_trace(go.Scatter(x=[x0], y=[f0 / 1e6], mode="markers+text",
                             marker=dict(color=WARN, size=11, symbol="diamond"),
                             text=[f"  mean offset {x0:.0f} m @ {f0/1e6:.1f} MN"],
                             textposition="middle right", textfont=dict(size=11, color=INK),
                             name="rated-thrust mean offset"))
    # Guide lines to the operating point.
    fig.add_shape(type="line", x0=x0, y0=0, x1=x0, y1=f0 / 1e6,
                  line=dict(color=WARN, width=1, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=f0 / 1e6, x1=x0, y1=f0 / 1e6,
                  line=dict(color=WARN, width=1, dash="dash"))
    # Surge natural-period annotation.
    fig.add_annotation(x=0.06 * x.max(), y=0.86 * f_lin.max(),
                       text=(f"surge stiffness anchored to rated thrust<br>"
                             f"T<sub>n,surge</sub> ≈ {platform.surge_natural_period_s:.0f} s"),
                       showarrow=False, align="left", xanchor="left",
                       font=dict(size=11, color=GOOD),
                       bgcolor="rgba(255,255,255,0.75)")

    fig = theme.plotly_layout(
        fig, title="Mooring horizontal restoring vs surge offset",
        height=380, xtitle="surge offset (m)",
        ytitle="horizontal restoring force (MN)", legend=True)
    fig.update_xaxes(range=[0, x.max()])
    fig.update_yaxes(range=[0, 1.05 * f_nl.max()])
    return fig


def _circle(fig: go.Figure, cx: float, cy: float, r: float, fill: str,
            line: str = dk.BP_LINE, width: float = 1.2) -> None:
    fig.add_shape(type="circle", x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                  line=dict(color=line, width=width), fillcolor=fill)


def _beam(fig: go.Figure, x0: float, y0: float, x1: float, y1: float, width: float,
          fill: str) -> None:
    """Rectangular pontoon beam of the given width from (x0,y0) to (x1,y1)."""
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) + 1e-9
    nx, ny = -dy / L * width / 2.0, dx / L * width / 2.0
    fig.add_trace(go.Scatter(
        x=[x0 + nx, x1 + nx, x1 - nx, x0 - nx, x0 + nx],
        y=[y0 + ny, y1 + ny, y1 - ny, y0 - ny, y0 + ny],
        mode="lines", fill="toself", fillcolor=fill,
        line=dict(color=dk.BP_LINE, width=1.0), hoverinfo="skip", showlegend=False))


def _schedule_table(fig: go.Figure, x0: float, y0: float, col_w, row_h: float,
                    header, rows) -> None:
    """A compact drawn table (header + rows) anchored with its top-left at (x0, y0)."""
    ncol = len(col_w)
    w = sum(col_w)
    nrow = len(rows) + 1
    y1 = y0 - nrow * row_h
    fig.add_shape(type="rect", x0=x0, y0=y1, x1=x0 + w, y1=y0,
                  line=dict(color=dk.BP_LINE, width=1.2), fillcolor="#FFFFFF")
    # header separator + row lines
    fig.add_shape(type="line", x0=x0, y0=y0 - row_h, x1=x0 + w, y1=y0 - row_h,
                  line=dict(color=dk.BP_LINE, width=1.0))
    for k in range(2, nrow):
        yy = y0 - k * row_h
        fig.add_shape(type="line", x0=x0, y0=yy, x1=x0 + w, y1=yy,
                      line=dict(color=dk.BP_LINE, width=0.4))
    cx = x0
    for c in col_w[:-1]:
        cx += c
        fig.add_shape(type="line", x0=cx, y0=y1, x1=cx, y1=y0,
                      line=dict(color=dk.BP_LINE, width=0.5))
    for j, htxt in enumerate(header):
        hx = x0 + sum(col_w[:j]) + col_w[j] / 2.0
        fig.add_annotation(x=hx, y=y0 - row_h / 2.0, text=f"<b>{htxt}</b>", showarrow=False,
                           font=dict(size=8, color=dk.BP_LINE))
    for i, row in enumerate(rows):
        ry = y0 - (i + 1.5) * row_h
        for j, val in enumerate(row):
            vx = x0 + sum(col_w[:j]) + col_w[j] / 2.0
            fig.add_annotation(x=vx, y=ry, text=str(val), showarrow=False,
                               font=dict(size=8, color=INK))


def mooring_plan(platform: Platform = VOLTURNUS_S) -> go.Figure:
    """Top-down mooring layout: 3-line spread, watch circles, anchors, schedule table."""
    R = platform.column_spacing_m
    Rc = platform.column_diameter_m / 2.0
    angs = (0.0, 120.0, 240.0)
    offsets = [(R * math.cos(math.radians(a)), R * math.sin(math.radians(a))) for a in angs]
    moor_draw = 232.0                          # foreshortened drawn anchor radius
    mean_off = platform.static_surge_at_rated_m           # 20 m
    watch = 0.05 * platform.water_depth_m + mean_off      # representative max watch radius

    xr = (-268.0, 272.0)
    yr = (-268.0, 268.0)
    dx = xr[1] - xr[0]
    fig = go.Figure()
    dk.blueprint_axes(fig, xr, yr, height=640, equal=True)

    # --- watch circles (mean offset + max excursion) ----------------------
    th = np.linspace(0.0, 2.0 * np.pi, 180)
    for rad, col, lab in ((mean_off, ACCENT, f"mean offset {mean_off:.0f} m"),
                          (watch, WARN, f"watch circle ≈{watch:.0f} m (repr.)")):
        fig.add_trace(go.Scatter(x=rad * np.cos(th), y=rad * np.sin(th), mode="lines",
                                 line=dict(color=col, width=1.1, dash="dot"),
                                 hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=rad * math.cos(math.radians(45)),
                           y=rad * math.sin(math.radians(45)) + 60.0, text=lab,
                           showarrow=True, arrowhead=0, arrowcolor=col, arrowwidth=0.6,
                           ax=0, ay=-18, font=dict(size=8, color=col),
                           bgcolor="rgba(244,247,250,0.85)")

    # --- column pitch circle + hull footprint -----------------------------
    fig.add_trace(go.Scatter(x=R * np.cos(th), y=R * np.sin(th), mode="lines",
                             line=dict(color=dk.BP_LINE, width=0.7, dash="dashdot"),
                             hoverinfo="skip", showlegend=False))
    for (ox, oy) in offsets:
        _beam(fig, 0.0, 0.0, ox, oy, width=9.0, fill=dk.STEEL_D)
    for (ox, oy) in offsets:
        _circle(fig, ox, oy, Rc, dk.STEEL)
    _circle(fig, 0.0, 0.0, Rc, dk.STEEL)

    # --- 3-line mooring spread --------------------------------------------
    for k, (a, (ox, oy)) in enumerate(zip(angs, offsets), start=1):
        ux, uy = math.cos(math.radians(a)), math.sin(math.radians(a))
        fx, fy = ox + ux * (Rc + 1.5), oy + uy * (Rc + 1.5)     # fairlead
        nx, ny = -uy, ux
        # fairlead delta plate
        fig.add_trace(go.Scatter(
            x=[fx + ux * 8, fx + nx * 4.5, fx - nx * 4.5, fx + ux * 8],
            y=[fy + uy * 8, fy + ny * 4.5, fy - ny * 4.5, fy + uy * 8],
            mode="lines", fill="toself", fillcolor=dk.STEEL_D,
            line=dict(color=dk.BP_LINE, width=1.0), hoverinfo="skip", showlegend=False))
        axp, ayp = ux * moor_draw, uy * moor_draw
        fig.add_shape(type="line", x0=fx, y0=fy, x1=axp, y1=ayp,
                      line=dict(color=dk.STEEL, width=1.6))
        # break mark (foreshortening)
        bx, by = 0.70 * moor_draw * ux, 0.70 * moor_draw * uy
        fig.add_shape(type="line", x0=bx - 5 * nx - 4 * ux, y0=by - 5 * ny - 4 * uy,
                      x1=bx + 5 * nx + 4 * ux, y1=by + 5 * ny + 4 * uy,
                      line=dict(color=dk.BP_INK, width=1.0))
        _draw_anchor(fig, axp, ayp)
        fig.add_annotation(x=axp + 15.0 * ux, y=ayp + 15.0 * uy, text=f"<b>MA{k}</b>",
                           showarrow=False, font=dict(size=9, color=dk.BP_INK),
                           bgcolor="rgba(244,247,250,0.85)")
        # line label near the fairlead
        fig.add_annotation(x=fx + ux * 46, y=fy + uy * 46, text=f"ML{k}", showarrow=False,
                           font=dict(size=10, color=dk.BP_INK), bgcolor="rgba(244,247,250,0.85)")

    # --- dimensions -------------------------------------------------------
    dk.dim_radial(fig, 0.0, 0.0, R, ang_deg=90.0, text=f"{R:.2f} m")
    dk.dim_angular(fig, 0.0, 0.0, R * 0.55, a0_deg=0.0, a1_deg=120.0, text="120°")
    dk.dim_angular(fig, 0.0, 0.0, R * 0.55, a0_deg=120.0, a1_deg=240.0, text="120°")
    # anchor-radius dimension along ML1 (label lands mid-line, clear of centre)
    fig.add_shape(type="line", x0=Rc, y0=0.0, x1=moor_draw, y1=0.0,
                  line=dict(color=dk.BP_LINE, width=0.7, dash="dot"))
    fig.add_annotation(x=0.5 * moor_draw, y=0.0, text=f"R ≈ {_ANCHOR_RADIUS_M:.0f} m (repr.)",
                       showarrow=False, yshift=9, font=dict(size=8, color=dk.BP_INK),
                       bgcolor="rgba(244,247,250,0.85)")

    # --- mooring schedule table (lower-left) ------------------------------
    rows = [(f"ML{k}", f"{a:.0f}°", f"{_PRETENSION_MN:.1f} MN",
             f"{_TOP_CHAIN['mbl_mn']:.0f} MN") for k, a in enumerate(angs, start=1)]
    _schedule_table(fig, x0=-42.0, y0=yr[0] + 118.0,
                    col_w=[30.0, 30.0, 44.0, 34.0], row_h=16.0,
                    header=("LINE", "BRG", "PRETEN.", "MBL"), rows=rows)

    # --- furniture --------------------------------------------------------
    dk.north_arrow(fig, x=xr[0] + 0.10 * dx, y=yr[1] - 0.20 * (yr[1] - yr[0]), size=34.0)
    dk.frame(fig, xr, yr, zones=True)
    dk.notes_block(fig, xr, yr, [
        "Plan; dimensions in metres. 120° three-line spread.",
        "Chain – polyester – chain make-up (see HAL-MOR-001).",
        "Lines foreshortened to a break — real R ≈800 m (repr.).",
        "Pretension / MBL representative (ASSUMPTIONS.md).",
    ], corner="top-right")
    dk.revision_table(fig, xr, yr, [("A", "issued — representative spread")])
    dk.title_block(fig, xr, yr, title="MOORING LAYOUT — PLAN",
                   subtitle="UMaine VolturnUS-S · 3 × catenary lines, 120° spread",
                   dwg_no="HAL-MOR-002", scale="NTS", rev="A",
                   extra="computed layout · R foreshortened")
    dk.scale_bar(fig, x=xr[0] + 26.0, y=yr[0] + 26.0, length_m=100.0, n=4, unit="m")
    return fig
