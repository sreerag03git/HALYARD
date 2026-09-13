"""2-D technical drawings — dimensioned, from computed geometry (§8).

  * lazy-wave profile from the solved static shape (hang-off, sag, hog, touchdown,
    buoyancy section) with dimensions;
  * cable cross-section schematic with the equivalent-stress (armour) layer labelled;
  * system-coupling block diagram (electrical -> rotor -> thrust -> platform -> cable -> fatigue).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from app import theme
from app.theme import ACCENT, GREY, INK, WARN, GOOD, GRID_LINE
from models.dynamic_cable import DynamicCable
from physics.cable import StaticShape, _region_indices

CABLE_BLACK = "#20242A"
BUOY = "#E0A03A"


def lazywave_profile(shape: StaticShape, cable: DynamicCable) -> go.Figure:
    reg = _region_indices(shape, cable)
    depth = cable.water_depth_m
    fig = go.Figure()
    # Seabed and MSL.
    fig.add_hrect(y0=-depth - 6, y1=-depth, fillcolor="#D8CBB0", line_width=0, layer="below")
    fig.add_hline(y=0.0, line=dict(color="#7FA8C9", width=1.5, dash="dot"),
                  annotation_text="MSL", annotation_position="top left")
    fig.add_hline(y=-depth, line=dict(color="#B0A080", width=1))
    # Buoyancy section highlighted.
    L = cable.total_length_m
    b0, b1 = cable.buoyancy_start_frac * L, cable.buoyancy_end_frac * L
    bmask = (shape.s >= b0) & (shape.s <= b1)
    fig.add_trace(go.Scatter(x=shape.x, y=shape.z, mode="lines",
                             line=dict(color=ACCENT, width=3), name="cable"))
    fig.add_trace(go.Scatter(x=shape.x[bmask], y=shape.z[bmask], mode="lines",
                             line=dict(color=WARN, width=6), name="buoyancy section"))
    # Region markers.
    for name, i in reg.items():
        fig.add_trace(go.Scatter(x=[shape.x[i]], y=[shape.z[i]], mode="markers+text",
                                 marker=dict(color=INK, size=8), text=[name.replace("_", "-")],
                                 textposition="top center", showlegend=False,
                                 textfont=dict(size=10)))
    # Dimension annotations.
    fig.add_annotation(x=shape.x[reg["touchdown"]] / 2, y=6,
                       text=f"hang-off → touchdown ≈ {shape.x[reg['touchdown']]:.0f} m",
                       showarrow=False, font=dict(size=10, color=GREY))
    fig.add_annotation(x=shape.x.max() * 0.02, y=-depth / 2,
                       text=f"water depth {depth:.0f} m", showarrow=False,
                       font=dict(size=10, color=GREY), textangle=-90)
    fig = theme.plotly_layout(fig, "Lazy-wave dynamic-cable profile (solved static shape)",
                              xtitle="horizontal distance (m)", ytitle="elevation (m)")
    fig.update_yaxes(scaleanchor=None)
    return fig


def cross_section(cable: DynamicCable) -> go.Figure:
    """Datasheet-quality cable cross-section: 3 cores in trefoil, layers, armour wires."""
    fig = go.Figure()
    Rmm = cable.outer_diameter_m * 500                       # outer radius [mm]
    th = np.linspace(0, 2 * np.pi, 120)

    def ring(r_out, r_in, color, opacity=1.0, line="#3A3A38"):
        xo, yo = r_out * np.cos(th), r_out * np.sin(th)
        xi, yi = r_in * np.cos(th[::-1]), r_in * np.sin(th[::-1])
        fig.add_trace(go.Scatter(x=np.concatenate([xo, xi]), y=np.concatenate([yo, yi]),
                                 mode="lines", fill="toself", fillcolor=color, opacity=opacity,
                                 line=dict(color=line, width=1), hoverinfo="skip",
                                 showlegend=False))

    # Concentric construction (outer serving -> armour bedding -> armour -> inner sheath -> fillers).
    ring(Rmm, Rmm - 6, "#3B4046")                            # outer serving
    ring(Rmm - 6, Rmm - 10, "#8A8A86")                       # armour bedding
    r_arm = Rmm - 14
    ring(Rmm - 10, r_arm - 4, "#C9C9C4")                     # armour annulus (base)
    ring(r_arm - 4, r_arm - 8, "#8A8A86")                    # inner bedding
    ring(r_arm - 8, 0, "#B9975B", 0.35)                      # filler / binder

    # Two counter-helical armour layers drawn as discrete round wires (galvanized steel).
    for rr, nwire, wr in ((r_arm - 1, 42, 2.4), (Rmm - 11.5, 46, 2.4)):
        ang = np.linspace(0, 2 * np.pi, nwire, endpoint=False)
        for a in ang:
            cx, cy = rr * np.cos(a), rr * np.sin(a)
            fig.add_trace(go.Scatter(x=cx + wr * np.cos(th), y=cy + wr * np.sin(th),
                                     mode="lines", fill="toself", fillcolor=WARN,
                                     line=dict(color="#7A5A1E", width=0.6), opacity=0.95,
                                     hoverinfo="skip", showlegend=False))

    # Three power cores in a trefoil (conductor, XLPE insulation, screen).
    core_r = (r_arm - 10) * 0.55
    core_off = (r_arm - 10) * 0.42
    for a in (90, 210, 330):
        cx, cy = core_off * np.cos(np.radians(a)), core_off * np.sin(np.radians(a))
        for rad, col in ((core_r, "#5A6B7A"), (core_r * 0.78, "#D8D2C4"),
                         (core_r * 0.52, "#B87333")):
            fig.add_trace(go.Scatter(x=cx + rad * np.cos(th), y=cy + rad * np.sin(th),
                                     mode="lines", fill="toself", fillcolor=col,
                                     line=dict(color="#3A3A38", width=0.8), hoverinfo="skip",
                                     showlegend=False))

    # Callouts.
    labels = [(0, Rmm - 3, "outer serving"), (r_arm, 0, "armour wires (stress layer)"),
              (core_off * np.cos(np.radians(90)), core_off * np.sin(np.radians(90)),
               "Cu conductor + XLPE"), (0, -Rmm * 0.45, "fillers")]
    for lx, ly, tx in labels:
        fig.add_annotation(x=lx, y=ly, ax=lx + Rmm * 0.9, ay=ly - Rmm * 0.5, xref="x", yref="y",
                           axref="x", ayref="y", text=tx, showarrow=True, arrowhead=2,
                           arrowsize=1, arrowcolor=INK, font=dict(size=9, color=INK))
    # Outer-diameter dimension.
    fig.add_annotation(x=0, y=-Rmm - 8, text=f"⌀ {cable.outer_diameter_m*1000:.0f} mm  ·  "
                       f"66 kV 3-core dynamic  ·  armour r={cable.armour_wire_radius_m*1000:.1f} mm "
                       f"(slip), pitch R={cable.armour_pitch_radius_m*1000:.0f} mm (stick)",
                       showarrow=False, font=dict(size=9, color=GREY))
    fig = theme.plotly_layout(fig, "Dynamic cable cross-section (to scale)",
                              xtitle="mm", ytitle="mm", legend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(range=[-Rmm - 15, Rmm + 15])
    fig.update_yaxes(range=[-Rmm - 18, Rmm + 12])
    return fig


def coupling_block_diagram() -> go.Figure:
    """The causation chain as a labelled block diagram (electrical -> ... -> fatigue)."""
    steps = [
        ("Grid frequency\n(swing eq.)", ACCENT),
        ("Converter power\ncommand ΔP", ACCENT),
        ("Generator torque\n→ rotor speed ω", ACCENT),
        ("Aero thrust\nCₜ(λ)", WARN),
        ("Platform motion\nsurge + pitch", WARN),
        ("Cable curvature\n& tension", GOOD),
        ("Hang-off\nfatigue", INK),
    ]
    fig = go.Figure()
    n = len(steps)
    bw, gap = 1.3, 0.55            # box width and gap (wide boxes, room for text + arrows)
    pitch = bw + gap
    for i, (label, color) in enumerate(steps):
        x0 = i * pitch
        fig.add_shape(type="rect", x0=x0, x1=x0 + bw, y0=0, y1=1, layer="below",
                      line=dict(color=color, width=2), fillcolor="rgba(255,255,255,0.95)")
        fig.add_annotation(x=x0 + bw / 2, y=0.5, text=label.replace("\n", "<br>"),
                           showarrow=False, font=dict(size=11, color=INK), align="center")
        if i < n - 1:
            fig.add_annotation(x=x0 + bw + gap * 0.9, y=0.5, ax=x0 + bw + gap * 0.1, ay=0.5,
                               xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowcolor=GREY, arrowwidth=1.6, text="")
    # Recovery-lever callout spanning the rotor -> thrust -> platform links.
    fig.add_annotation(x=3.5 * pitch - gap, y=1.35,
                       text="recovery shape = the only design lever (§5.3)",
                       showarrow=False, font=dict(size=11, color=ACCENT))
    fig.update_xaxes(visible=False, range=[-0.3, n * pitch - gap + 0.3])
    fig.update_yaxes(visible=False, range=[-0.25, 1.7])
    fig.update_layout(height=170, margin=dict(l=0, r=0, t=8, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Blueprint helpers (drawing frame, title block, dimension lines)
# ---------------------------------------------------------------------------
_BP_BG = "#F4F7FA"
_BP_LINE = "#37506B"


def _blueprint_frame(fig, xr, yr, title, subtitle, scale_note=""):
    fig.add_shape(type="rect", x0=xr[0], y0=yr[0], x1=xr[1], y1=yr[1],
                  line=dict(color=_BP_LINE, width=2), fillcolor=_BP_BG, layer="below")
    fig.add_shape(type="rect", x0=xr[0] + (xr[1] - xr[0]) * 0.006,
                  y0=yr[0] + (yr[1] - yr[0]) * 0.01,
                  x1=xr[1] - (xr[1] - xr[0]) * 0.006, y1=yr[1] - (yr[1] - yr[0]) * 0.01,
                  line=dict(color=_BP_LINE, width=1), layer="below")
    # Title block (bottom-right).
    bw, bh = (xr[1] - xr[0]) * 0.34, (yr[1] - yr[0]) * 0.13
    x0, y0 = xr[1] - bw - (xr[1] - xr[0]) * 0.01, yr[0] + (yr[1] - yr[0]) * 0.012
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x0 + bw, y1=y0 + bh,
                  line=dict(color=_BP_LINE, width=1.5), fillcolor="white")
    fig.add_annotation(x=x0 + bw / 2, y=y0 + bh * 0.66, text=f"<b>{title}</b>",
                       showarrow=False, font=dict(size=12, color=INK))
    fig.add_annotation(x=x0 + bw / 2, y=y0 + bh * 0.36, text=subtitle, showarrow=False,
                       font=dict(size=9, color=GREY))
    fig.add_annotation(x=x0 + bw / 2, y=y0 + bh * 0.12,
                       text=f"HALYARD · {scale_note} · computed geometry", showarrow=False,
                       font=dict(size=8, color=GREY))


def _dim(fig, p1, p2, text, off=0.0, horiz=True, color=_BP_LINE):
    """Dimension line with arrowheads + extension lines and a measurement label."""
    (x1, y1), (x2, y2) = p1, p2
    if horiz:
        yl = max(y1, y2) + off
        fig.add_shape(type="line", x0=x1, y0=y1, x1=x1, y1=yl, line=dict(color=color, width=0.8))
        fig.add_shape(type="line", x0=x2, y0=y2, x1=x2, y1=yl, line=dict(color=color, width=0.8))
        fig.add_annotation(x=x2, y=yl, ax=x1, ay=yl, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1, arrowcolor=color)
        fig.add_annotation(x=x1, y=yl, ax=x2, ay=yl, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1, arrowcolor=color)
        fig.add_annotation(x=(x1 + x2) / 2, y=yl, text=text, showarrow=False, yshift=8,
                           font=dict(size=9, color=INK))
    else:
        xl = max(x1, x2) + off
        fig.add_shape(type="line", x0=x1, y0=y1, x1=xl, y1=y1, line=dict(color=color, width=0.8))
        fig.add_shape(type="line", x0=x2, y0=y2, x1=xl, y1=y2, line=dict(color=color, width=0.8))
        fig.add_annotation(x=xl, y=y2, ax=xl, ay=y1, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1, arrowcolor=color)
        fig.add_annotation(x=xl, y=y1, ax=xl, ay=y2, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1, arrowcolor=color)
        fig.add_annotation(x=xl, y=(y1 + y2) / 2, text=text, showarrow=False, xshift=6,
                           textangle=-90, font=dict(size=9, color=INK))


def general_arrangement(shape: StaticShape, cable: DynamicCable, platform) -> go.Figure:
    """Dimensioned side-elevation general-arrangement drawing (blueprint style)."""
    depth = platform.water_depth_m
    reg = _region_indices(shape, cable)
    xr = (-160, shape.x.max() + 90)
    yr = (-depth - 45, platform.hub_height_m + 160)
    fig = go.Figure()
    _blueprint_frame(fig, xr, yr, "GENERAL ARRANGEMENT — ELEVATION",
                     "IEA-15MW · UMaine VolturnUS-S · lazy-wave dynamic cable",
                     "NTS")
    # Waterline + seabed hatch.
    fig.add_shape(type="line", x0=xr[0] + 10, y0=0, x1=xr[1] - 10, y1=0,
                  line=dict(color="#5C7FA3", width=1.2, dash="dashdot"))
    fig.add_annotation(x=xr[0] + 24, y=4, text="MSL", showarrow=False,
                       font=dict(size=9, color="#5C7FA3"))
    fig.add_shape(type="rect", x0=xr[0] + 10, y0=-depth - 8, x1=xr[1] - 10, y1=-depth,
                  fillcolor="#D9CDB0", line=dict(color="#B7A57F", width=1))
    for xx in np.linspace(xr[0] + 20, xr[1] - 20, 26):
        fig.add_shape(type="line", x0=xx, y0=-depth, x1=xx - 6, y1=-depth - 8,
                      line=dict(color="#B7A57F", width=0.6))
    # Turbine elevation (tower + nacelle + rotor circle + 3 blades).
    fb, hub = platform.freeboard_m, platform.hub_height_m
    fig.add_shape(type="rect", x0=-3, y0=fb, x1=3, y1=hub, fillcolor="#9AA7B4",
                  line=dict(color=_BP_LINE, width=1))
    th = np.linspace(0, 2 * np.pi, 80)
    fig.add_trace(go.Scatter(x=120 * np.cos(th), y=hub + 120 * np.sin(th), mode="lines",
                             line=dict(color=_BP_LINE, width=1), hoverinfo="skip",
                             showlegend=False))
    for a in (90, 210, 330):
        fig.add_shape(type="line", x0=0, y0=hub, x1=115 * np.cos(np.radians(a)),
                      y1=hub + 115 * np.sin(np.radians(a)), line=dict(color=_BP_LINE, width=2.5))
    # Platform columns (front + rear) as rectangles.
    for xx in (-platform.column_spacing_m / 2, 0, platform.column_spacing_m / 2):
        fig.add_shape(type="rect", x0=xx - 6.25, y0=-platform.draft_m, x1=xx + 6.25, y1=fb,
                      fillcolor="#AEBAC6", line=dict(color=_BP_LINE, width=1))
    fig.add_shape(type="rect", x0=-platform.column_spacing_m / 2 - 6, y0=-platform.draft_m,
                  x1=platform.column_spacing_m / 2 + 6, y1=-platform.draft_m + 7,
                  fillcolor="#93A0AD", line=dict(color=_BP_LINE, width=1))
    # Cable profile + buoyancy band.
    L = cable.total_length_m
    b0, b1 = cable.buoyancy_start_frac * L, cable.buoyancy_end_frac * L
    bmask = (shape.s >= b0) & (shape.s <= b1)
    fig.add_trace(go.Scatter(x=shape.x, y=shape.z, mode="lines",
                             line=dict(color=CABLE_BLACK, width=2.5), name="dynamic cable",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=shape.x[bmask], y=shape.z[bmask], mode="lines",
                             line=dict(color=BUOY, width=6), name="buoyancy section",
                             hoverinfo="skip"))
    for name, i in reg.items():
        fig.add_annotation(x=shape.x[i], y=shape.z[i], text=name.replace("_", "-"),
                           showarrow=True, arrowhead=2, arrowsize=1, arrowcolor=GREY,
                           ax=0, ay=-24, font=dict(size=9, color=INK))
    # Dimensions.
    _dim(fig, (0, hub), (0, 0), "hub 150 m", off=0, horiz=False, color=_BP_LINE)
    _dim(fig, (-120, hub), (120, hub), "rotor Ø 240 m", off=18, horiz=True)
    _dim(fig, (shape.x[1], -depth - 18), (shape.x[reg['touchdown']], -depth - 18),
         f"hang-off → touchdown {shape.x[reg['touchdown']]:.0f} m", off=0, horiz=True)
    _dim(fig, (xr[0] + 30, 0), (xr[0] + 30, -depth), f"{depth:.0f} m", off=0, horiz=False)
    fig = theme.plotly_layout(fig, "", height=560, xtitle="", ytitle="", legend=True)
    fig.update_xaxes(visible=False, range=xr)
    fig.update_yaxes(visible=False, range=yr, scaleanchor="x", scaleratio=1)
    fig.update_layout(plot_bgcolor=_BP_BG, paper_bgcolor="rgba(0,0,0,0)")
    return fig
