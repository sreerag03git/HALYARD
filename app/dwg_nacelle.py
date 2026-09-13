"""Rotor-nacelle assembly (RNA) drivetrain cutaway — longitudinal section (§8).

A dimensioned blueprint cutaway of the IEA-15MW direct-drive (Type-4) nacelle: yaw
bearing on the tower top, the main frame / bedplate, a single main bearing, the large-
diameter direct-drive permanent-magnet generator (outer rotor ring + PM poles + inner
stator — NO gearbox), the hub with three pitch bearings (blade roots shown in section),
spinner nose cone and the nacelle cover envelope. The section is taken along the shaft
axis (drawn horizontal, standard practice; the 6-deg shaft tilt is called out). Rating /
rotor numbers are the frozen turbine values; drivetrain dimensions are representative
published IEA-15-240-RWT values, disclosed on the drawing, and drawn to scale.

Built only from geometry; Plotly only; no streamlit import.
"""
from __future__ import annotations

import math

import plotly.graph_objects as go

from app import dwg_kit as dk
from app.theme import INK
from models.iea15mw import IEA15MW, Turbine

# --- representative IEA-15-240-RWT drivetrain geometry [m] (disclosed) ------
_TILT_DEG = 6.0                 # rotor shaft tilt (published)
_CONE_DEG = 4.0                 # blade precone (published)
_TOWER_TOP_DIA = 6.5            # tower-top / yaw-bearing outer diameter
_YAW_H = 1.3                    # yaw-bearing + main-frame base height
_H_AXIS = 6.6                   # shaft-axis height above the tower-top flange
_GEN_AIRGAP_DIA = 10.5          # direct-drive generator air-gap diameter (headline)
_GEN_LEN = 2.4                  # generator axial (stack) length
_GEN_RING = 0.6                 # radial thickness of rotor / stator active rings
_HUB_DIA = 7.94                 # hub diameter
_HUB_LEN = 4.6                  # hub axial length
_SPINNER_LEN = 3.8              # spinner nose cone length beyond the hub
_S_GEN = 1.1                    # generator centre, axial station (upwind +)
_S_MAINBRG = 3.6                # main-bearing station
_S_HUB = 7.3                    # hub-centre station
_BLADE_ROOT_CHORD = 5.4         # blade-root chord (root cylinder diameter)
_BLADE_ROOT_SHOW = 5.2          # radial length of the root stub drawn in section

_COVER = "#9AAAB8"             # nacelle cover
_FRAME = "#5E6B78"             # bedplate / main frame
_ROTOR = "#6C7B89"             # generator rotor ring
_STATOR = "#B9C2CB"            # generator stator ring
_MAGNET = "#37506B"            # PM poles
_HUBC = "#7C8B99"              # hub casting


def _poly(fig, pts, color, line=dk.BP_LINE, width=1.0, opacity=1.0):
    xs = [p[0] for p in pts] + [pts[0][0]]
    zs = [p[1] for p in pts] + [pts[0][1]]
    fig.add_trace(go.Scatter(x=xs, y=zs, mode="lines", fill="toself", fillcolor=color,
                             line=dict(color=line, width=width), opacity=opacity,
                             hoverinfo="skip", showlegend=False))


def _ring_section(fig, o, s0, s1, r_in, r_out, color, line=dk.BP_LINE):
    """Two cut faces (upper + lower) of an axisymmetric ring in longitudinal section."""
    for sgn in (+1, -1):
        pts = [(o[0] + s0, o[1] + sgn * r_in), (o[0] + s1, o[1] + sgn * r_in),
               (o[0] + s1, o[1] + sgn * r_out), (o[0] + s0, o[1] + sgn * r_out)]
        _poly(fig, pts, color, line=line, width=1.0)


def nacelle_cutaway(turbine: Turbine | None = None) -> go.Figure:
    """Dimensioned longitudinal cutaway of the direct-drive RNA (shaft axis horizontal)."""
    turbine = turbine or IEA15MW
    H = _YAW_H + _H_AXIS                   # shaft-axis height
    o = (0.0, H)
    r_go = _GEN_AIRGAP_DIA / 2.0           # generator outer radius
    r_hub = _HUB_DIA / 2.0

    xr = (-10.5, 16.5)
    yr = (-3.5, H + r_go + 8.5)
    fig = go.Figure()
    dk.blueprint_axes(fig, xr, yr, height=620, equal=True)

    # --- nacelle cover envelope (symmetric, tapered nose) -----------------
    cov = [(-6.2, H - (r_go + 1.4)), (5.8, H - (r_go + 1.4)), (6.7, H - (r_go - 1.6)),
           (6.7, H + (r_go - 1.6)), (5.8, H + (r_go + 1.4)), (-6.2, H + (r_go + 1.4))]
    _poly(fig, cov, _COVER, width=1.4, opacity=0.30)

    # --- tower top + yaw bearing ------------------------------------------
    fig.add_shape(type="rect", x0=-_TOWER_TOP_DIA / 2, y0=yr[0], x1=_TOWER_TOP_DIA / 2, y1=0.0,
                  line=dict(color=dk.BP_LINE, width=1.2), fillcolor=_FRAME)
    dk.hatch_band(fig, -_TOWER_TOP_DIA / 2, _TOWER_TOP_DIA / 2, yr[0], 0.0,
                  angle_deg=60, color=dk.SEABED_LINE, spacing=0.75)
    fig.add_shape(type="rect", x0=-_TOWER_TOP_DIA / 2 - 0.4, y0=0.0,
                  x1=_TOWER_TOP_DIA / 2 + 0.4, y1=_YAW_H,
                  line=dict(color=dk.BP_LINE, width=1.2), fillcolor=_STATOR)
    dk.centerline(fig, 0.0, yr[0], 0.0, _YAW_H + 1.2)
    dk.leader(fig, 0.0, _YAW_H, "yaw bearing + drives", dx=-7.5, dy=2.0, size=9)

    # --- bedplate / main frame (supports the generator + main bearing) ----
    bed = [(-5.8, H - r_go + 0.3), (4.4, H - r_go + 0.3), (4.4, H - r_go - 0.8),
           (2.4, _YAW_H), (-2.4, _YAW_H), (-5.8, H - r_go - 0.8)]
    _poly(fig, bed, _FRAME, width=1.2)

    # --- direct-drive generator: rotor ring, PM poles (2 air-gap bands), stator
    _ring_section(fig, o, _S_GEN - _GEN_LEN / 2, _S_GEN + _GEN_LEN / 2,
                  r_go - _GEN_RING, r_go, _ROTOR)                       # outer rotor ring
    for sgn in (+1, -1):                                               # PM poles on rotor bore
        rb = r_go - _GEN_RING
        npole = 7
        for k in range(npole):
            sx = _S_GEN - _GEN_LEN / 2 + 0.18 + k * (_GEN_LEN - 0.36) / npole
            _poly(fig, [(o[0] + sx, o[1] + sgn * rb),
                        (o[0] + sx + 0.18, o[1] + sgn * rb),
                        (o[0] + sx + 0.18, o[1] + sgn * (rb - 0.45)),
                        (o[0] + sx, o[1] + sgn * (rb - 0.45))], _MAGNET, line=_MAGNET, width=0.3)
    _ring_section(fig, o, _S_GEN - _GEN_LEN / 2 + 0.12, _S_GEN + _GEN_LEN / 2 - 0.12,
                  r_go - 2 * _GEN_RING - 0.5, r_go - _GEN_RING - 0.55, _STATOR)  # stator ring
    dk.leader(fig, o[0] + _S_GEN, o[1] + r_go - _GEN_RING / 2, "PM rotor ring",
              dx=2.4, dy=2.6, size=9)
    dk.leader(fig, o[0] + _S_GEN, o[1] + r_go - 1.7 * _GEN_RING,
              "stator — direct-drive, no gearbox", dx=-2.2, dy=3.2, size=9)

    # --- main shaft + single main bearing ---------------------------------
    _ring_section(fig, o, -0.6, _S_HUB - 0.4, 0.0, 0.8, _HUBC)          # shaft core (solid)
    _ring_section(fig, o, _S_MAINBRG - 0.5, _S_MAINBRG + 0.5, 0.8, 1.7, _STATOR)  # main bearing
    dk.leader(fig, o[0] + _S_MAINBRG, o[1] + 1.6, "main bearing", dx=1.6, dy=2.6, size=9)
    dk.centerline(fig, o[0] - 7.0, o[1], o[0] + _S_HUB + _SPINNER_LEN + 2.2, o[1])

    # --- hub (solid section) + spinner cone + 2 blade roots ---------------
    _ring_section(fig, o, _S_HUB - _HUB_LEN / 2, _S_HUB + _HUB_LEN / 2, 0.0, r_hub, _HUBC)
    _poly(fig, [(o[0] + _S_HUB + _HUB_LEN / 2, o[1] + r_hub),
                (o[0] + _S_HUB + _HUB_LEN / 2, o[1] - r_hub),
                (o[0] + _S_HUB + _HUB_LEN / 2 + _SPINNER_LEN, o[1])], _COVER, opacity=0.5)
    hw = _BLADE_ROOT_CHORD / 2.0
    for sgn in (+1, -1):                              # upper + lower blade roots in section
        y0 = o[1] + sgn * r_hub
        y1 = o[1] + sgn * (r_hub + _BLADE_ROOT_SHOW)
        _poly(fig, [(o[0] + _S_HUB - hw, y0), (o[0] + _S_HUB + hw, y0),
                    (o[0] + _S_HUB + hw, y1), (o[0] + _S_HUB - hw, y1)], _HUBC,
              width=1.0, opacity=0.9)
        fig.add_shape(type="circle", x0=o[0] + _S_HUB - 1.05, y0=y0 - 1.05,
                      x1=o[0] + _S_HUB + 1.05, y1=y0 + 1.05,
                      line=dict(color=dk.BP_LINE, width=1.0), fillcolor=_STATOR)
    dk.leader(fig, o[0] + _S_HUB + hw, o[1] + r_hub + _BLADE_ROOT_SHOW * 0.55,
              "blade root + pitch bearing", dx=1.5, dy=1.4, size=9)
    dk.leader(fig, o[0] + _S_HUB + _HUB_LEN / 2 + _SPINNER_LEN, o[1], "spinner",
              dx=1.2, dy=-2.6, size=9)
    dk.leader(fig, o[0] + _S_HUB, o[1] - r_hub * 0.4, "hub casting", dx=-2.4, dy=-3.4, size=9)

    # --- dimensions --------------------------------------------------------
    dk.dim_v(fig, o[1] - r_go, o[1] + r_go, xr[0] + 2.2,
             f"⌀ {_GEN_AIRGAP_DIA:.1f} m gen. air-gap", ext=abs((o[0] + _S_GEN) - (xr[0] + 2.2)))
    dk.dim_v(fig, o[1] - r_hub, o[1] + r_hub, xr[1] - 1.5, f"⌀ {_HUB_DIA:.2f} m hub",
             ext=-abs((xr[1] - 1.5) - (o[0] + _S_HUB)))
    dk.dim_h(fig, cov[0][0], o[0] + _S_HUB + _HUB_LEN / 2 + _SPINNER_LEN, yr[1] - 1.7,
             "nacelle + hub ≈ 20 m", ext=-3.0)

    # --- shaft-tilt callout (axis drawn horizontal) -----------------------
    tx = -8.5
    fig.add_shape(type="line", x0=tx, y0=_YAW_H + 1.0, x1=tx + 4.2, y1=_YAW_H + 1.0,
                  line=dict(color=dk.BP_LINE, width=0.7, dash="dot"))
    fig.add_shape(type="line", x0=tx, y0=_YAW_H + 1.0,
                  x1=tx + 4.2, y1=_YAW_H + 1.0 + 4.2 * math.tan(math.radians(_TILT_DEG)),
                  line=dict(color=dk.BP_LINE, width=1.0))
    dk.dim_angular(fig, tx, _YAW_H + 1.0, 3.4, 0.0, _TILT_DEG, f"{_TILT_DEG:.0f}° shaft tilt")

    dk.section_marker(fig, xr[0] + 3.2, yr[1] - 5.2, "A", angle_deg=0.0)

    # --- furniture ---------------------------------------------------------
    dk.frame(fig, xr, yr, zones=True)
    dk.notes_block(fig, xr, yr, [
        f"IEA-15-240-RWT — {turbine.rated_power_W/1e6:.0f} MW direct-drive (Type-4), rotor "
        f"⌀ {2*turbine.rotor_radius_m:.0f} m, {turbine.omega_rated_rads*60/(2*math.pi):.2f} rpm.",
        "Direct-drive PM generator: NO gearbox — large air-gap-diameter ring generator.",
        "Drivetrain dimensions REPRESENTATIVE (published IEA-15MW); section to scale.",
        f"Section on the shaft axis (drawn horizontal); shaft tilt {_TILT_DEG:.0f}°, "
        f"precone {_CONE_DEG:.0f}°.",
    ], corner="top-left")
    dk.revision_table(fig, xr, yr, [("A", "issued — representative drivetrain")])
    dk.title_block(fig, xr, yr, title="NACELLE / DRIVETRAIN — CUTAWAY SECTION A–A",
                   subtitle="IEA-15MW direct-drive RNA · hub height 150 m",
                   dwg_no="HAL-RNA-001", scale="1:NTS", rev="A",
                   extra="to-scale section (repr. drivetrain)")
    dk.scale_bar(fig, x=xr[0] + 3.2, y=yr[0] + 1.2, length_m=5.0, n=5, unit="m")
    return fig
