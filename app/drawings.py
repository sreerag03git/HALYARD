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
    """Concentric-layer schematic of the cable cross-section (dimensioned, labelled)."""
    fig = go.Figure()
    D = cable.outer_diameter_m
    layers = [
        (D / 2, "#5A5A55", "outer sheath"),
        (cable.armour_pitch_radius_m + 0.006, "#8A8A86", "bedding"),
        (cable.armour_pitch_radius_m, WARN, "armour (stress layer)"),
        (cable.armour_pitch_radius_m - 0.012, "#B9975B", "insulation / cores"),
        (0.028, ACCENT, "3 conductor cores"),
    ]
    th = np.linspace(0, 2 * np.pi, 80)
    for radius, color, label in layers:
        fig.add_trace(go.Scatter(x=radius * np.cos(th) * 1000, y=radius * np.sin(th) * 1000,
                                 mode="lines", fill="toself", line=dict(color=color, width=1.5),
                                 fillcolor=color, opacity=0.65, name=label))
    # Armour wire radius callout.
    fig.add_annotation(x=cable.armour_pitch_radius_m * 1000, y=0,
                       ax=cable.armour_pitch_radius_m * 1000 + 40, ay=-30,
                       text=f"armour pitch R={cable.armour_pitch_radius_m*1000:.0f} mm<br>"
                            f"wire r={cable.armour_wire_radius_m*1000:.1f} mm",
                       showarrow=True, arrowcolor=INK, font=dict(size=10))
    fig = theme.plotly_layout(fig, f"Cable cross-section (OD {D*1000:.0f} mm) — equivalent-stress layer",
                              xtitle="mm", ytitle="mm")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
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
