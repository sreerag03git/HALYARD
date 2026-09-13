"""Datasheet-grade dynamic-cable engineering drawings (§8) — blueprint style.

Three CAD-quality 2-D drawings built procedurally from the real ``DynamicCable``
numbers and the solved lazy-wave static shape, drafted with the shared
``app.dwg_kit`` furniture (frame, title block, dimensions, leaders, hatching):

  * ``cross_section_datasheet`` — true-scale 66 kV three-core section with two
    counter-helical armour layers, trefoil power cores and a component schedule
    (supersedes ``app.drawings.cross_section``);
  * ``lazywave_config`` — the solved lazy-wave configuration with buoyancy-module
    schedule, region callouts and dimensions (upgrades ``lazywave_profile``);
  * ``bend_stiffener_detail`` — DETAIL A of the hang-off / bell-mouth / moulded
    polyurethane bend stiffener.

Headless (no streamlit); every public function returns a ``go.Figure`` built from
the real objects and is independently callable with the defaults shown.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from app import dwg_kit as kit
from app.dwg_kit import (BP_INK, BP_LINE, BUOY, CABLE_BLACK, SEABED, SEABED_LINE,
                         STEEL, STEEL_D)
from app.theme import ACCENT, GOOD, GREY, INK, WARN
from models.dynamic_cable import REFERENCE_CABLE, DynamicCable
from physics.cable import (BEND_STIFFENER_LEN_M, StaticShape, _region_indices,
                           solve_static_shape)

# Cross-section material palette (datasheet greys + copper) -----------------
_C_SERVING = "#3B4046"    # PP-yarn outer serving
_C_BEDDING = "#6E7A85"    # armour bedding / inter-layer tapes
_C_SHEATH = "#4A5560"     # inner sheath
_C_FILLER = "#CBBE9A"     # polymer filler / binder
_C_ARM_OUT = "#95A2AF"    # outer armour wire (galv. steel)
_C_ARM_IN = "#6E7C8A"     # inner armour wire (opposite lay — darker shade)
_C_ARM_OUT_L = "#4E5A66"
_C_ARM_IN_L = "#3B444E"
_C_PE = "#2E3238"         # core PE oversheath
_C_LEAD = "#A6ACB2"       # lead-alloy metallic sheath
_C_SEMI = "#383836"       # semiconductive screens
_C_XLPE = "#ECE6D6"       # XLPE insulation
_C_CU = "#B5732E"         # stranded copper conductor
_C_CU_STRAND = "#C98A45"


# --- shared primitive ------------------------------------------------------
def _ring(fig: go.Figure, r_out: float, r_in: float, color: str, cx: float = 0.0,
          cy: float = 0.0, line: str = "#2A2E33", width: float = 0.8,
          opacity: float = 1.0) -> None:
    """Filled concentric ring (or disk when ``r_in``<=0) centred on (cx, cy)."""
    th = np.linspace(0.0, 2 * np.pi, 160)
    xo, yo = cx + r_out * np.cos(th), cy + r_out * np.sin(th)
    if r_in <= 0.0:
        x, y = xo, yo
    else:
        xi, yi = cx + r_in * np.cos(th[::-1]), cy + r_in * np.sin(th[::-1])
        x, y = np.concatenate([xo, xi]), np.concatenate([yo, yi])
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=color,
                             opacity=opacity, line=dict(color=line, width=width),
                             hoverinfo="skip", showlegend=False))


def _disc(fig: go.Figure, cx: float, cy: float, r: float, color: str,
          line: str = "#2A2E33", width: float = 0.8, opacity: float = 1.0) -> None:
    """Small filled circle (a wire / rod / strand)."""
    th = np.linspace(0.0, 2 * np.pi, 40)
    fig.add_trace(go.Scatter(x=cx + r * np.cos(th), y=cy + r * np.sin(th), mode="lines",
                             fill="toself", fillcolor=color, opacity=opacity,
                             line=dict(color=line, width=width), hoverinfo="skip",
                             showlegend=False))


def _poly(fig: go.Figure, xs: Sequence[float], ys: Sequence[float], color: str,
          line: str = BP_LINE, width: float = 1.0, opacity: float = 1.0) -> None:
    """Filled closed polygon."""
    fig.add_trace(go.Scatter(x=list(xs) + [xs[0]], y=list(ys) + [ys[0]], mode="lines",
                             fill="toself", fillcolor=color, opacity=opacity,
                             line=dict(color=line, width=width), hoverinfo="skip",
                             showlegend=False))


def _schedule(fig: go.Figure, x0: float, y_top: float, w: float, rh: float,
              rows: Sequence[Tuple[str, str, str]], title: str = "COMPONENT SCHEDULE") -> None:
    """Small item/description/material table drawn from shapes + annotations."""
    n = len(rows)
    y_bot = y_top - rh * (n + 1)
    c1 = x0 + 0.14 * w        # ITEM column
    c2 = x0 + 0.52 * w        # COMPONENT column
    fig.add_shape(type="rect", x0=x0, y0=y_bot, x1=x0 + w, y1=y_top,
                  line=dict(color=BP_LINE, width=1.2), fillcolor="#FFFFFF")
    fig.add_shape(type="line", x0=x0, y0=y_top - rh, x1=x0 + w, y1=y_top - rh,
                  line=dict(color=BP_LINE, width=1.0))
    for cx in (c1, c2):
        fig.add_shape(type="line", x0=cx, y0=y_bot, x1=cx, y1=y_top,
                      line=dict(color=BP_LINE, width=0.6))
    pad = 0.02 * w
    hy = y_top - 0.5 * rh
    fig.add_annotation(x=(x0 + c1) / 2, y=hy, text="<b>#</b>", showarrow=False,
                       font=dict(size=8, color=BP_INK))
    fig.add_annotation(x=(c1 + c2) / 2, y=hy, text=f"<b>{title}</b>", showarrow=False,
                       font=dict(size=8, color=BP_INK))
    fig.add_annotation(x=(c2 + x0 + w) / 2, y=hy, text="<b>MATERIAL</b>", showarrow=False,
                       font=dict(size=8, color=BP_INK))
    for k, (item, comp, mat) in enumerate(rows):
        yy = y_top - rh - (k + 0.5) * rh
        if k:
            fig.add_shape(type="line", x0=x0, y0=yy + 0.5 * rh, x1=x0 + w, y1=yy + 0.5 * rh,
                          line=dict(color=BP_LINE, width=0.4))
        fig.add_annotation(x=(x0 + c1) / 2, y=yy, text=item, showarrow=False,
                           font=dict(size=8, color=BP_INK))
        fig.add_annotation(x=c1 + pad, y=yy, text=comp, showarrow=False, xanchor="left",
                           font=dict(size=8, color=INK))
        fig.add_annotation(x=c2 + pad, y=yy, text=mat, showarrow=False, xanchor="left",
                           font=dict(size=8, color=GREY))


# ===========================================================================
# 1. Cross-section datasheet
# ===========================================================================
def cross_section_datasheet(cable: DynamicCable) -> go.Figure:
    """True-scale 66 kV three-core dynamic-cable section (radius in mm).

    Concentric construction as filled rings — PP serving, bedding, two counter-
    helical galvanised-steel armour layers drawn as discrete round wires (opposite
    lay by shade + tangential lay tick), inner sheath, polymer filler — with three
    power cores in a trefoil (stranded Cu / semicon screens / XLPE / lead sheath /
    PE oversheath), a fibre-optic unit and filler rods in the interstices. Callouts,
    a DIA-200 / armour-radii dimension set and a component schedule. Supersedes
    ``app.drawings.cross_section``.
    """
    R = cable.outer_diameter_m * 1000.0 / 2.0          # outer radius [mm] = 100
    r_pitch = cable.armour_pitch_radius_m * 1000.0     # 90 mm (stick lever)
    r_wire = cable.armour_wire_radius_m * 1000.0       # 2.5 mm (slip lever)

    xr = (-330.0, 340.0)
    yr = (-300.0, 300.0)
    fig = go.Figure()
    kit.blueprint_axes(fig, xr, yr, height=640, equal=True)

    # --- concentric construction (outer -> in) -----------------------------
    _ring(fig, R, 0.94 * R, _C_SERVING)                    # PP serving 100->94
    _ring(fig, 0.94 * R, 0.78 * R, _C_BEDDING)             # armour bedding band 94->78
    _ring(fig, 0.78 * R, 0.73 * R, _C_SHEATH)              # inner sheath 78->73
    _ring(fig, 0.73 * R, 0.0, _C_FILLER)                   # polymer filler core disk

    # Two counter-helical armour layers as discrete round wires + lay ticks.
    for (rr, nwire, fillc, linec, lay) in ((r_pitch, 46, _C_ARM_OUT, _C_ARM_OUT_L, +1),
                                           (0.82 * R, 42, _C_ARM_IN, _C_ARM_IN_L, -1)):
        ang = np.linspace(0.0, 2 * np.pi, nwire, endpoint=False)
        for a in ang:
            cx, cy = rr * math.cos(a), rr * math.sin(a)
            _disc(fig, cx, cy, r_wire, fillc, line=linec, width=0.5)
            # short tangential tick showing lay direction (opposite for the two layers)
            tx, ty = -math.sin(a), math.cos(a)
            fig.add_shape(type="line", x0=cx - lay * 0.9 * r_wire * tx,
                          y0=cy - lay * 0.9 * r_wire * ty,
                          x1=cx + lay * 0.9 * r_wire * tx,
                          y1=cy + lay * 0.9 * r_wire * ty,
                          line=dict(color=linec, width=0.6))

    # --- three power cores in a trefoil ------------------------------------
    annulus = 0.73 * R
    offset = 0.52 * annulus
    core = 0.45 * annulus                                   # PE-oversheath radius
    core_angles = (90.0, 210.0, 330.0)
    for a in core_angles:
        cx, cy = offset * math.cos(math.radians(a)), offset * math.sin(math.radians(a))
        _ring(fig, core, 0.94 * core, _C_PE, cx, cy)                 # PE oversheath
        _ring(fig, 0.94 * core, 0.88 * core, _C_LEAD, cx, cy)        # lead-alloy sheath
        _ring(fig, 0.88 * core, 0.85 * core, _C_SEMI, cx, cy)        # insulation screen
        _ring(fig, 0.85 * core, 0.53 * core, _C_XLPE, cx, cy)        # XLPE insulation
        _ring(fig, 0.53 * core, 0.48 * core, _C_SEMI, cx, cy)        # conductor screen
        _ring(fig, 0.48 * core, 0.0, _C_CU, cx, cy, line="#7A4A1E")  # stranded Cu (fill)
        # stranded conductor detail — a 1+6+12 packing of round strands
        srad = 0.13 * core
        _disc(fig, cx, cy, srad, _C_CU_STRAND, line="#7A4A1E", width=0.4)
        for ring_i, nk in ((1, 6), (2, 12)):
            for k in range(nk):
                ak = 2 * math.pi * k / nk
                rk = ring_i * 2.05 * srad
                _disc(fig, cx + rk * math.cos(ak), cy + rk * math.sin(ak), srad,
                      _C_CU_STRAND, line="#7A4A1E", width=0.4)

    # --- interstitial fillers + fibre-optic unit ---------------------------
    r_inter = offset * 0.98
    for a in (30.0, 150.0, 270.0):
        ix, iy = r_inter * math.cos(math.radians(a)), r_inter * math.sin(math.radians(a))
        if a == 270.0:                                      # fibre-optic unit
            _disc(fig, ix, iy, 0.11 * annulus, "#26506E", line=ACCENT, width=0.8)
            _disc(fig, ix, iy, 0.05 * annulus, BUOY, line="#7A5A1E", width=0.6)
        else:                                               # polymer filler rods
            _disc(fig, ix, iy, 0.11 * annulus, "#B8A986", line="#8A7C58", width=0.6)

    # --- dimensions & callouts ---------------------------------------------
    kit.dim_h(fig, -R, R, -R - 42, f"⌀ {cable.outer_diameter_m*1000:.0f} mm", ext=-14)
    kit.dim_radial(fig, 0, 0, r_pitch, 52, f"{r_pitch:.0f} (pitch, stick)")
    kit.centerline(fig, -R - 12, 0, R + 12, 0)
    kit.centerline(fig, 0, -R - 12, 0, R + 12)

    lead = kit.leader
    lead(fig, 0.0, 0.97 * R, "PP-yarn outer serving", dx=-150, dy=70)
    lead(fig, r_pitch * math.cos(math.radians(62)), r_pitch * math.sin(math.radians(62)),
         "armour — 2 counter-helical<br>galv.-steel layers (stress/fatigue)", dx=95, dy=95)
    lead(fig, 0.82 * R * math.cos(math.radians(158)), 0.82 * R * math.sin(math.radians(158)),
         f"inner armour (opposite lay)<br>wire r = {r_wire:.1f} mm (slip)", dx=-120, dy=40)
    lead(fig, 0.755 * R * math.cos(math.radians(200)), 0.755 * R * math.sin(math.radians(200)),
         "inner sheath + bedding", dx=-150, dy=-20)
    # core-internal callouts led to the lower-right core (330 deg)
    ccx = offset * math.cos(math.radians(330))
    ccy = offset * math.sin(math.radians(330))
    lead(fig, ccx + 0.97 * core, ccy, "PE oversheath", dx=120, dy=40)
    lead(fig, ccx + 0.90 * core * math.cos(math.radians(-30)),
         ccy + 0.90 * core * math.sin(math.radians(-30)), "lead-alloy sheath", dx=130, dy=-8)
    lead(fig, ccx + 0.70 * core, ccy, "XLPE insulation", dx=120, dy=-45)
    lead(fig, ccx + 0.50 * core * math.cos(math.radians(-55)),
         ccy + 0.50 * core * math.sin(math.radians(-55)), "semicon screens", dx=95, dy=-95)
    lead(fig, ccx, ccy, "stranded Cu conductor", dx=60, dy=-120)
    fx, fy = r_inter * math.cos(math.radians(270)), r_inter * math.sin(math.radians(270))
    lead(fig, fx, fy, "fibre-optic unit", dx=-95, dy=-95, color=ACCENT)
    ffx, ffy = r_inter * math.cos(math.radians(150)), r_inter * math.sin(math.radians(150))
    lead(fig, ffx, ffy, "polymer filler", dx=-110, dy=25)

    # --- schedule, notes, frame, title block -------------------------------
    _schedule(fig, -312, 262, 176, 26,
              [("1", "Conductor", "Stranded Cu"),
               ("2", "Screens", "Semiconductive"),
               ("3", "Insulation", "XLPE"),
               ("4", "Core sheath", "Lead alloy"),
               ("5", "Armour", "Galv. steel ×2"),
               ("6", "Serving", "PP yarn"),
               ("7", "Filler / FO", "Polymer / SM fibre")])
    kit.notes_block(fig, xr, yr,
                    [f"Section TO SCALE (mm); overall ⌀ {cable.outer_diameter_m*1000:.0f} mm.",
                     "Armour lever: r = 2.5 mm (slip) ≤ used ≤ R = 90 mm (stick).",
                     "3-core 66 kV; one FO unit, polymer fillers in interstices."],
                    corner="top-right")
    kit.frame(fig, xr, yr)
    kit.scale_bar(fig, -300, -258, 100.0, n=4, unit="mm")
    kit.revision_table(fig, xr, yr, [("A", "computed section")])
    kit.title_block(fig, xr, yr, "DYNAMIC CABLE — CROSS SECTION",
                    subtitle="66 kV 3-core lazy-wave dynamic export cable",
                    dwg_no="HAL-CBL-001", scale="TO SCALE (mm)", rev="A",
                    extra="procedural — real DynamicCable")
    return fig


# ===========================================================================
# 2. Lazy-wave configuration
# ===========================================================================
def lazywave_config(shape: StaticShape, cable: DynamicCable) -> go.Figure:
    """Solved lazy-wave configuration drawing (buoyancy schedule + dimensions).

    The suspended profile ``shape.x``/``shape.z`` (from ``solve_static_shape``) with
    the buoyancy section highlighted and its module schedule, the four critical
    regions labelled by leader, and a full dimension set: water depth, hang-off
    elevation, hog/sag elevations, hang-off→touchdown span, total length and the
    MBR / peak-curvature check. Upgrades ``app.drawings.lazywave_profile``.
    """
    reg = _region_indices(shape, cable)
    depth = cable.water_depth_m
    L = cable.total_length_m
    x, z = np.asarray(shape.x), np.asarray(shape.z)
    x_td = float(x[reg["touchdown"]])
    z_hog = float(z[reg["hog"]])
    z_sag = float(z[reg["sag"]])

    x_max = float(x.max())
    xr = (-46.0, x_max + 62.0)
    yr = (-depth - 40.0, 58.0)
    fig = go.Figure()
    kit.blueprint_axes(fig, xr, yr, height=620, equal=True)

    # --- environment: seabed hatch + waterline -----------------------------
    kit.hatch_band(fig, xr[0] + 6, xr[1] - 6, -depth - 30, -depth, angle_deg=45,
                   spacing=9.0, color=SEABED_LINE, fill=SEABED)
    fig.add_shape(type="line", x0=xr[0] + 6, y0=-depth, x1=xr[1] - 6, y1=-depth,
                  line=dict(color=SEABED_LINE, width=1.4))
    kit.waterline(fig, xr[0] + 6, xr[1] - 6, z=0.0)

    # --- suspended cable + buoyancy section --------------------------------
    b0, b1 = cable.buoyancy_start_frac * L, cable.buoyancy_end_frac * L
    bmask = (shape.s >= b0) & (shape.s <= b1)
    fig.add_trace(go.Scatter(x=x, y=z, mode="lines", line=dict(color=CABLE_BLACK, width=2.6),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x[bmask], y=z[bmask], mode="lines",
                             line=dict(color=BUOY, width=7), hoverinfo="skip",
                             showlegend=False))
    # discrete buoyancy-module ticks along the section (schematic count)
    bidx = np.where(bmask)[0]
    if len(bidx) > 2:
        for j in bidx[1:-1:2]:
            dx_, dz_ = x[j + 1] - x[j - 1], z[j + 1] - z[j - 1]
            nlen = math.hypot(dx_, dz_) + 1e-9
            nx, nz = -dz_ / nlen, dx_ / nlen
            fig.add_shape(type="line", x0=x[j] - 3.2 * nx, y0=z[j] - 3.2 * nz,
                          x1=x[j] + 3.2 * nx, y1=z[j] + 3.2 * nz,
                          line=dict(color="#9A6B1E", width=2.0))

    # --- hang-off + bend-stiffener flag ------------------------------------
    x_ho, z_ho = cable.hangoff_x_m, cable.hangoff_z_m
    fig.add_shape(type="rect", x0=x_ho - 5, y0=z_ho, x1=x_ho + 5, y1=0.0,
                  fillcolor=STEEL, line=dict(color=BP_LINE, width=1.0))     # riser porch stub
    _disc(fig, x_ho, z_ho, 2.0, WARN, line=BP_INK, width=1.0)
    kit.leader(fig, x_ho, z_ho, "HANG-OFF (bend stiffener)<br>SEE DETAIL A — HAL-CBL-003",
               dx=-30, dy=34, color=BP_INK)

    # --- region callouts ----------------------------------------------------
    pretty = {"hang_off": "HANG-OFF", "sag": "SAG", "hog": "HOG (crest)",
              "touchdown": "TOUCHDOWN"}
    offs = {"hang_off": (-26, 18), "sag": (-30, -30), "hog": (10, 34),
            "touchdown": (34, 22)}
    for name in ("sag", "hog", "touchdown"):
        i = reg[name]
        dx_, dy_ = offs[name]
        kit.leader(fig, float(x[i]), float(z[i]), pretty[name], dx=dx_, dy=dy_, color=ACCENT)
        _disc(fig, float(x[i]), float(z[i]), 1.6, ACCENT, line=BP_INK, width=0.8)

    # --- dimensions ---------------------------------------------------------
    kit.dim_v(fig, -depth, 0.0, xr[0] + 22, f"water depth {depth:.0f} m", ext=-12)
    kit.dim_v(fig, z_ho, 0.0, x_ho - 18, f"hang-off {abs(z_ho):.0f} m", ext=10)
    kit.dim_h(fig, x_ho, x_td, -depth - 20, f"hang-off → touchdown {x_td:.0f} m", ext=-10)
    # hog / sag elevation leaders (read from the solved shape)
    kit.leader(fig, float(x[reg["hog"]]), z_hog, f"hog elev {z_hog:.1f} m",
               dx=26, dy=-16, color=GOOD)
    kit.leader(fig, float(x[reg["sag"]]), z_sag, f"sag elev {z_sag:.1f} m",
               dx=-8, dy=-30, color=WARN)

    # --- curvature check & buoyancy schedule --------------------------------
    kappa_pk = float(np.max(shape.curvature[: reg["touchdown"] + 1]))
    mbr_pk = (1.0 / kappa_pk) if kappa_pk > 1e-9 else float("inf")
    k_lim = cable.curvature_limit_1_per_m
    n_mod = int(round((b1 - b0) / 2.5))
    kit.notes_block(fig, xr, yr,
                    [f"Total cable length {L:.0f} m; hang-off {x_ho:.0f} m to subsea anchor.",
                     f"Buoyancy modules: ≈{n_mod} @ 2.5 m pitch over "
                     f"{b0:.0f}–{b1:.0f} m ({cable.buoyancy_start_frac*100:.0f}–"
                     f"{cable.buoyancy_end_frac*100:.0f}% L); net uplift "
                     f"{cable.buoyancy_uplift_factor:.1f}× submerged wt.",
                     f"MBR {cable.min_bend_radius_m:.1f} m → κ_limit {k_lim:.3f} 1/m; "
                     f"peak computed κ {kappa_pk:.3f} 1/m (R {mbr_pk:.1f} m) — "
                     f"{'OK' if kappa_pk <= k_lim else 'REVIEW'}.",
                     "True scale (H:V = 1:1)."],
                    corner="top-right")
    kit.frame(fig, xr, yr)
    kit.scale_bar(fig, xr[0] + 20, -depth - 26, 50.0, n=5, unit="m")
    kit.revision_table(fig, xr, yr, [("A", "solved static shape")])
    kit.title_block(fig, xr, yr, "DYNAMIC CABLE — LAZY-WAVE CONFIGURATION",
                    subtitle="solved static shape · buoyancy hog/sag decoupling",
                    dwg_no="HAL-CBL-002", scale="1:1 (TRUE)", rev="A",
                    extra="procedural — solve_static_shape()")
    return fig


# ===========================================================================
# 3. Hang-off & bend-stiffener detail (DETAIL A)
# ===========================================================================
def bend_stiffener_detail(cable: DynamicCable) -> go.Figure:
    """DETAIL A — hang-off I/J-tube bell mouth + moulded PU bend stiffener.

    Zoomed true-scale detail of the hang-off: the steel bell mouth, the tapered
    polyurethane bend-stiffener cone (length ``BEND_STIFFENER_LEN_M`` = 4 m) clamped
    to the porch, and the cable curving through it to its far-field departure angle
    (from the cheaply solved ``shape.departure_angle_deg``). Annotates the stiffener
    length, the MBR curvature limit, the departure angle and the hang-off stick note.
    """
    shape = solve_static_shape(cable)                 # cheap single solve for the angle
    alpha = float(shape.departure_angle_deg)          # from vertical [deg]
    alpha_draw = max(6.0, min(alpha, 30.0))           # legible angle for the cone geometry
    a_rad = math.radians(alpha_draw)
    Ls = BEND_STIFFENER_LEN_M                         # 4 m
    r_cab = cable.outer_diameter_m / 2.0              # 0.10 m
    r_top = 0.35                                       # clamped stiffener root radius [m]

    # Cable centreline: clamped vertical at the bell mouth, easing to alpha over the
    # stiffener length, then straight — integrate theta(s) (from vertical).
    n = 60
    s = np.linspace(0.0, 1.6 * Ls, n)
    frac = np.clip(s / Ls, 0.0, 1.0)
    theta = a_rad * (3 * frac ** 2 - 2 * frac ** 3)   # smoothstep 0 -> alpha over [0,Ls]
    ds = np.gradient(s)
    cx = np.cumsum(np.sin(theta) * ds)
    cz = -np.cumsum(np.cos(theta) * ds)
    cx -= cx[0]
    cz -= cz[0]

    xr = (-1.6, 4.4)
    yr = (-7.4, 1.7)
    fig = go.Figure()
    kit.blueprint_axes(fig, xr, yr, height=620, equal=True)
    kit.waterline(fig, xr[0] + 0.2, xr[1] - 0.2, z=0.0, label="MSL (ref)")

    # --- I/J-tube bell mouth (steel, hatched) ------------------------------
    # vertical tube walls above the porch, flaring to a trumpet at the exit
    tube_w = r_top + 0.10
    top_y = 1.3
    porch_y = 0.0
    bell_y = -0.55
    for side in (-1, 1):
        xs = [side * tube_w, side * tube_w, side * (tube_w + 0.28), side * (r_top + 0.06),
              side * (r_top + 0.06)]
        ys = [top_y, bell_y + 0.30, bell_y, bell_y - 0.10, porch_y - 0.02]
        # steel wall as a thin hatched strip
        wx = [v + side * 0.10 for v in xs]
        kit.hatch_band(fig, min(side * tube_w, side * (tube_w + 0.28)),
                       max(side * tube_w, side * (tube_w + 0.28)),
                       bell_y, top_y, angle_deg=45, spacing=0.22, color=STEEL_D)
        _poly(fig, xs + wx[::-1], ys + ys[::-1], STEEL, line=STEEL_D, width=1.0, opacity=0.9)
    # porch / clamp plate
    fig.add_shape(type="rect", x0=-(r_top + 0.28), y0=-0.10, x1=(r_top + 0.28), y1=0.14,
                  fillcolor=STEEL_D, line=dict(color=BP_LINE, width=1.0))

    # --- moulded PU bend stiffener cone ------------------------------------
    # outer radius tapers from r_top at s=0 to ~r_cab at s=Ls, offset normal to centreline
    inb = s <= Ls
    ss, cxx, czz, th = s[inb], cx[inb], cz[inb], theta[inb]
    rad = r_top + (r_cab - r_top) * np.clip(ss / Ls, 0, 1)
    nx, nz = np.cos(th), np.sin(th)            # unit normal to centreline (points +x side)
    lx, lz = cxx - rad * nx, czz - rad * nz
    rx, rz = cxx + rad * nx, czz + rad * nz
    _poly(fig, np.concatenate([lx, rx[::-1]]), np.concatenate([lz, rz[::-1]]),
          "#E6B24A", line="#9A6B1E", width=1.2, opacity=0.85)

    # --- cable through & below the stiffener --------------------------------
    ex, ez = -r_cab * np.cos(theta), -r_cab * np.sin(theta)
    _poly(fig, np.concatenate([cx + ex, (cx - ex)[::-1]]),
          np.concatenate([cz + ez, (cz - ez)[::-1]]), CABLE_BLACK, line="#000000",
          width=0.8, opacity=1.0)
    kit.centerline(fig, 0.0, 0.15, cx[-1], cz[-1])
    # vertical reference for the departure angle
    kit.centerline(fig, 0.0, cz[np.argmax(s >= Ls)] , 0.0, cz[np.argmax(s >= Ls)] - 1.4)

    # --- dimensions & annotations ------------------------------------------
    bot_i = int(np.argmax(s >= Ls))
    kit.dim_v(fig, cz[bot_i], 0.0, xr[0] + 0.55, f"stiffener L = {Ls:.0f} m", ext=0.15)
    kit.dim_angular(fig, 0.0, cz[bot_i], 1.3, -90.0, -90.0 + alpha_draw,
                    f"α ≈ {alpha:.1f}° (from vert.)")
    kit.leader(fig, 0.0, top_y - 0.25, "I/J-TUBE BELL MOUTH", dx=1.9, dy=0.5)
    kit.leader(fig, rx[len(rx) // 2], rz[len(rz) // 2],
               "MOULDED PU BEND STIFFENER", dx=1.5, dy=0.6, color="#9A6B1E")
    kit.leader(fig, cx[-1], cz[-1], f"66 kV DYNAMIC CABLE ⌀{cable.outer_diameter_m*1000:.0f}",
               dx=1.6, dy=0.2, color=BP_INK)
    kit.section_marker(fig, 0.0, 1.45, "A")

    kit.notes_block(fig, xr, yr,
                    [f"Stiffener controls curvature ≤ 1/MBR = "
                     f"{cable.curvature_limit_1_per_m:.3f} 1/m (MBR "
                     f"{cable.min_bend_radius_m:.1f} m).",
                     f"Computed hang-off departure α = {alpha:.1f}° from vertical.",
                     f"HANG-OFF carries highest tension → armour STICKS (no slip): "
                     f"bending lever = armour pitch R {cable.armour_pitch_radius_m*1000:.0f} mm.",
                     "Detail TO SCALE (m)."],
                    corner="top-left")
    kit.frame(fig, xr, yr)
    kit.scale_bar(fig, xr[0] + 0.4, yr[0] + 0.5, 2.0, n=4, unit="m")
    kit.revision_table(fig, xr, yr, [("A", "hang-off detail")])
    kit.title_block(fig, xr, yr, "HANG-OFF & BEND STIFFENER — DETAIL A",
                    subtitle="I/J-tube bell mouth · moulded PU stiffener",
                    dwg_no="HAL-CBL-003", scale="1:1 (TRUE)", rev="A",
                    extra="procedural — solve_static_shape()")
    return fig
