"""Station-keeping mooring drawings — catenary line make-up + restoring law (§8).

Two engineering figures built from the frozen VolturnUS-S reference numbers:

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
