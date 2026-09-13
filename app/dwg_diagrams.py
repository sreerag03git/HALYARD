"""Signal-flow / P&ID-style block diagrams (§8) — the causation chain, the
aero-servo control loop, the 6-DOF Cummins coupled hydro system and the
end-to-end analysis pipeline.

Every figure is a genuine engineering schematic — dimensioned boxes, summing
junctions, directed arrows (with real feedback loops) and decision diamonds —
composed from the shared drafting kit (:mod:`app.dwg_kit`) and drawn from the
real project objects (solved cable shape, platform periods, controller gains).
Pure, headless, Plotly-only; no streamlit.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import plotly.graph_objects as go

from app import dwg_kit as kit
from app.dwg_kit import BP_INK, BP_LINE
from app.theme import ACCENT, GOOD, GREY, INK, SURFACE, WARN
from models.dynamic_cable import REFERENCE_CABLE
from models.iea15mw import IEA15MW
from models.volturnus_s import VOLTURNUS_S
from physics.cable import solve_static_shape

# Block palette -------------------------------------------------------------
_FILL = "#FFFFFF"            # near-white block fill
_FILL_SOFT = "#F7FAFC"       # tinted fill for grouped/aux blocks
_LEVER = WARN                # design-lever accent
_BAND = "rgba(180,99,44,0.10)"   # lever band tint


# --- primitive furniture ---------------------------------------------------
def _sheet(title: str, subtitle: str, dwg_no: str,
           xr: Sequence[float], yr: Sequence[float], height: int,
           notes: Optional[Sequence[str]] = None,
           notes_corner: str = "top-left") -> go.Figure:
    """A framed drawing sheet with title/revision blocks and optional notes."""
    fig = go.Figure()
    kit.blueprint_axes(fig, xr, yr, height=height, equal=True)
    kit.frame(fig, xr, yr, zones=True)
    kit.title_block(fig, xr, yr, title=title, subtitle=subtitle, dwg_no=dwg_no,
                    scale="NTS", rev="B", extra="signal-flow schematic")
    kit.revision_table(fig, xr, yr, rows=[("B", "block diagram issued")])
    if notes:
        kit.notes_block(fig, xr, yr, notes, corner=notes_corner)
    return fig


def _round_rect_path(x0: float, y0: float, x1: float, y1: float, r: float) -> str:
    """SVG path (Q-corners) for a rounded rectangle in data coordinates."""
    r = min(r, 0.5 * abs(x1 - x0), 0.5 * abs(y1 - y0))
    return (
        f"M {x0 + r},{y0} L {x1 - r},{y0} Q {x1},{y0} {x1},{y0 + r} "
        f"L {x1},{y1 - r} Q {x1},{y1} {x1 - r},{y1} "
        f"L {x0 + r},{y1} Q {x0},{y1} {x0},{y1 - r} "
        f"L {x0},{y0 + r} Q {x0},{y0} {x0 + r},{y0} Z"
    )


def _block(fig: go.Figure, cx: float, cy: float, w: float, h: float,
           title: str, sub: str = "", color: str = ACCENT,
           fill: str = _FILL, tsize: int = 11, ssize: int = 8,
           lever: bool = False) -> Tuple[float, float, float, float]:
    """Rounded process block with a bold title + small sub-label; returns bbox."""
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    fig.add_shape(type="path", path=_round_rect_path(x0, y0, x1, y1, min(w, h) * 0.16),
                  line=dict(color=color, width=2.4 if lever else 1.8), fillcolor=fill)
    ty = cy + (0.14 * h if sub else 0.0)
    fig.add_annotation(x=cx, y=ty, text=f"<b>{title}</b>", showarrow=False,
                       font=dict(size=tsize, color=INK), align="center")
    if sub:
        fig.add_annotation(x=cx, y=cy - 0.26 * h, text=sub, showarrow=False,
                           font=dict(size=ssize, color=GREY), align="center")
    return x0, y0, x1, y1


def _diamond(fig: go.Figure, cx: float, cy: float, w: float, h: float,
             title: str, sub: str = "", color: str = _LEVER) -> None:
    """Decision diamond with a bold label."""
    path = (f"M {cx},{cy + h / 2} L {cx + w / 2},{cy} "
            f"L {cx},{cy - h / 2} L {cx - w / 2},{cy} Z")
    fig.add_shape(type="path", path=path, line=dict(color=color, width=2.0),
                  fillcolor=_FILL_SOFT)
    fig.add_annotation(x=cx, y=cy + (0.10 * h if sub else 0.0),
                       text=f"<b>{title}</b>", showarrow=False,
                       font=dict(size=10, color=INK), align="center")
    if sub:
        fig.add_annotation(x=cx, y=cy - 0.22 * h, text=sub, showarrow=False,
                           font=dict(size=8, color=GREY), align="center")


def _sum(fig: go.Figure, cx: float, cy: float, r: float,
         labels: Sequence[Tuple[str, str]] = ()) -> None:
    """Summing junction (circle + cross); labels = list of (sign, position)."""
    fig.add_shape(type="circle", x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                  line=dict(color=BP_INK, width=1.6), fillcolor=_FILL)
    k = 0.62 * r
    fig.add_shape(type="line", x0=cx - k, y0=cy, x1=cx + k, y1=cy,
                  line=dict(color=BP_INK, width=1.0))
    fig.add_shape(type="line", x0=cx, y0=cy - k, x1=cx, y1=cy + k,
                  line=dict(color=BP_INK, width=1.0))
    off = 1.7 * r
    pos = {"left": (cx - off, cy + off * 0.6), "right": (cx + off, cy + off * 0.6),
           "top": (cx + off * 0.6, cy + off), "bottom": (cx + off * 0.6, cy - off)}
    for sign, where in labels:
        px, py = pos.get(where, (cx + off, cy + off))
        fig.add_annotation(x=px, y=py, text=f"<b>{sign}</b>", showarrow=False,
                           font=dict(size=12, color=BP_INK))


def _arrow(fig: go.Figure, x0: float, y0: float, x1: float, y1: float,
           color: str = BP_INK, width: float = 1.6, text: str = "",
           tpos: str = "mid", dash: Optional[str] = None) -> None:
    """Straight directed connector head at (x1,y1); optional signal label."""
    if dash:
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=color, width=width, dash=dash))
        _head(fig, x1, y1, math.degrees(math.atan2(y1 - y0, x1 - x0)), color)
    else:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                           axref="x", ayref="y", text="", showarrow=True,
                           arrowhead=3, arrowsize=1.2, arrowwidth=width, arrowcolor=color)
    if text:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if tpos == "above":
            my += 2.6
        elif tpos == "below":
            my -= 2.6
        fig.add_annotation(x=mx, y=my, text=text, showarrow=False,
                           font=dict(size=8, color=BP_INK),
                           bgcolor="rgba(244,247,250,0.85)")


def _head(fig: go.Figure, x: float, y: float, ang_deg: float, color: str = BP_INK) -> None:
    """A single arrowhead landing at (x,y) pointing along ang_deg (data heading)."""
    a = math.radians(ang_deg)
    fig.add_annotation(x=x, y=y, ax=-14 * math.cos(a), ay=14 * math.sin(a),
                       xref="x", yref="y", axref="pixel", ayref="pixel", text="",
                       showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.6,
                       arrowcolor=color)


def _poly(fig: go.Figure, pts: Sequence[Tuple[float, float]], color: str = BP_INK,
          width: float = 1.4, dash: Optional[str] = None, text: str = "") -> None:
    """Orthogonal/feedback polyline with a terminal arrowhead."""
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        fig.add_shape(type="line", x0=ax, y0=ay, x1=bx, y1=by,
                      line=dict(color=color, width=width, dash=dash or "solid"))
    (px, py), (qx, qy) = pts[-2], pts[-1]
    _head(fig, qx, qy, math.degrees(math.atan2(qy - py, qx - px)), color)
    if text:
        mx, my = pts[0]
        fig.add_annotation(x=mx, y=my, text=text, showarrow=False, yshift=8,
                           font=dict(size=8, color=color),
                           bgcolor="rgba(244,247,250,0.85)")


# --- 1. coupling causation chain -------------------------------------------
def coupling_block() -> go.Figure:
    """Upgraded causation chain: grid frequency → … → hang-off fatigue → cost."""
    cable = REFERENCE_CABLE
    shape = solve_static_shape(cable)          # real solve (used for annotation)
    kappa = cable.curvature_limit_1_per_m
    dep = shape.departure_angle_deg

    xr, yr = (0.0, 176.0), (0.0, 96.0)
    fig = _sheet("SYSTEM COUPLING — CAUSATION CHAIN",
                 "grid-frequency support → cable hang-off fatigue → cost",
                 "HAL-BLK-01", xr, yr, height=560,
                 notes=[
                     "Directed arrows = physical causation; lower loop = floating feedback.",
                     f"Curvature limit 1/MBR = {kappa:.3f} m⁻¹ (MBR {cable.min_bend_radius_m:.1f} m).",
                     f"Solved hang-off departure angle {dep:.1f}° (static shape).",
                 ], notes_corner="top-left")

    stages = [
        ("Grid frequency", "swing eq.  df/dt", ACCENT),
        ("Support power ΔP", "synth-inertia H + droop R", ACCENT),
        ("Generator torque", "Type-4 converter (η 0.95)", ACCENT),
        ("Rotor speed ω", f"J = {IEA15MW.rotor_inertia_kgm2:.2e} kg·m²", ACCENT),
        ("Aero thrust", "Cₜ(λ, β) — IEA-15MW", WARN),
        ("Platform 6-DOF", "surge + pitch", WARN),
        ("Cable curvature\n& tension", "lazy-wave hang-off", GOOD),
        ("Hang-off fatigue", "rainflow + S-N (C203)", GOOD),
        ("Fatigue COST", "annualised €/yr", INK),
    ]
    n = len(stages)
    bw, bh = 14.5, 15.0
    x0, xN = 12.0, 164.0
    cy = 60.0
    xs = [x0 + (xN - x0) * i / (n - 1) for i in range(n)]
    for (title, sub, col), cx in zip(stages, xs):
        _block(fig, cx, cy, bw, bh, title.replace("\n", "<br>"), sub, color=col)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(fig, a + bw / 2, cy, b - bw / 2, cy, color=BP_LINE, width=1.8)

    # signal labels on selected links
    _arrow(fig, xs[3] + bw / 2, cy, xs[4] - bw / 2, cy, color=BP_LINE, width=1.8,
           text="ω", tpos="above")
    _arrow(fig, xs[5] + bw / 2, cy, xs[6] - bw / 2, cy, color=BP_LINE, width=1.8,
           text="surge, pitch", tpos="above")

    # SHAPED-RECOVERY lever band spanning torque → thrust → platform (stages 3..6)
    lx0, lx1 = xs[2] - bw / 2 - 1.5, xs[5] + bw / 2 + 1.5
    ly0, ly1 = cy - bh / 2 - 2.0, cy + bh / 2 + 10.0
    fig.add_shape(type="rect", x0=lx0, y0=ly0, x1=lx1, y1=ly1, layer="below",
                  line=dict(color=_LEVER, width=1.4, dash="dot"), fillcolor=_BAND)
    fig.add_annotation(x=(lx0 + lx1) / 2, y=ly1 - 2.6,
                       text="<b>SHAPED RECOVERY — the sole design lever</b>  (τ_rec)",
                       showarrow=False, font=dict(size=11, color=_LEVER))

    # floating-feedback loop: platform motion → nacelle-velocity pitch term → thrust
    fby = 40.0
    gx = 0.5 * (xs[3] + xs[4])
    _poly(fig, [(xs[5], cy - bh / 2), (xs[5], fby), (gx, fby)],
          color=ACCENT, width=1.5, dash="dot")
    _block(fig, gx, fby, 30.0, 9.5, "Floating feedback",
           "nacelle fore-aft velocity · K_F = 0.05", color=ACCENT, fill=_FILL_SOFT,
           tsize=10, ssize=8)
    _poly(fig, [(gx, fby + 4.75), (gx, cy - bh / 2)], color=ACCENT, width=1.5,
          dash="dot", text="Δβ → ΔCₜ")
    return fig


# --- 2. aero-servo control loop --------------------------------------------
def control_loop() -> go.Figure:
    """Closed-loop ROSCO-style aero-servo control block diagram."""
    om_r = IEA15MW.omega_rated_rads

    xr, yr = (0.0, 184.0), (0.0, 120.0)
    fig = _sheet("AERO-SERVO CONTROL — CLOSED LOOP",
                 "ROSCO-style blade-pitch + torque control with floating feedback",
                 "HAL-BLK-02", xr, yr, height=600,
                 notes=[
                     "Forward path (top) → speed; pitch loop (below) returns β.",
                     "Σ = summing junction;  → carries the named signal.",
                     f"ω_ref = rated {om_r:.3f} rad/s (7.56 rpm).",
                 ], notes_corner="top-right")

    top = 92.0                    # forward-path lane
    mid = 52.0                    # pitch-loop lane
    low = 24.0                    # grid-support lane

    # forward path: inflow → aero → Σ(T_aero−T_gen) → rotor inertia → ω
    _block(fig, 24, top, 28, 15, "Turbulent inflow", "Kaimal · IEC 61400-1", color=GREY)
    _block(fig, 68, top, 30, 15, "Rotor aerodynamics", "Cₚ/Cₜ(λ, β) — IEA-15MW deck", color=WARN)
    sx = 104.0
    _sum(fig, sx, top, 3.2, labels=[("+", "left"), ("−", "bottom")])
    _block(fig, 140, top, 30, 15, "Rotor inertia J", "∫  →  ω   (J = 3.16e8 kg·m²)", color=ACCENT)

    _arrow(fig, 38, top, 53, top, color=BP_LINE, text="wind V(t)", tpos="above")
    _arrow(fig, 83, top, sx - 3.2, top, color=BP_LINE, text="T_aero", tpos="above")
    _arrow(fig, sx + 3.2, top, 125, top, color=BP_LINE)
    _arrow(fig, 155, top, 172, top, color=ACCENT, width=2.0, text="ω", tpos="above")

    # generator / converter torque control (feeds −T_gen up into Σ)
    _block(fig, 104, low, 34, 14, "Generator torque control",
           "Type-4 full converter (η 0.95)", color=ACCENT, fill=_FILL_SOFT)
    _poly(fig, [(104, low + 7), (104, top - 3.2)], color=ACCENT, width=1.6,
          text="−T_gen")

    # grid-support path injecting ΔP into torque control
    _block(fig, 44, low, 34, 14, "Synthetic inertia + droop",
           "ΔP = 2H·df/dt + Δf/R  → τ_rec", color=ACCENT, fill=_FILL_SOFT)
    _arrow(fig, 61, low, 87, low, color=ACCENT, text="ΔP command", tpos="above")

    # speed feedback: ω → comparator with ω_ref
    cmpx = 172.0
    _sum(fig, cmpx, mid, 3.4, labels=[("−", "right"), ("+", "top")])
    _poly(fig, [(172, top - 4), (cmpx, mid + 3.4)], color=ACCENT, width=1.6)
    fig.add_annotation(x=cmpx + 7.5, y=mid + 6.5, text="ω_ref", showarrow=False,
                       font=dict(size=9, color=BP_INK))
    _arrow(fig, cmpx + 9, mid + 5, cmpx + 1.2, mid + 2.2, color=BP_LINE)

    # pitch loop (right→left): PI → Σ(+floating fb) → actuator+sat → inflow lag → β up to aero
    _block(fig, 138, mid, 30, 14, "Blade-pitch PI", "K_P = 6,  K_I = 2", color=GOOD)
    _sum(fig, 104, mid, 3.2, labels=[("+", "top"), ("+", "bottom")])
    _block(fig, 74, mid, 30, 14, "Pitch actuator", "τ = 0.10 s · β_max = 25°", color=GOOD)
    _block(fig, 34, mid, 30, 14, "Dynamic-inflow lag", "τ_ind = 4 s", color=WARN, fill=_FILL_SOFT)

    _arrow(fig, cmpx - 3.4, mid, 153, mid, color=BP_LINE, text="ω error", tpos="above")
    _arrow(fig, 123, mid, 107.2, mid, color=BP_LINE, text="β_cmd", tpos="above")
    _arrow(fig, 100.8, mid, 89, mid, color=BP_LINE)
    _arrow(fig, 59, mid, 49, mid, color=BP_LINE, text="β", tpos="above")
    # β returns up into the aero block
    _poly(fig, [(34, mid + 7), (34, 74), (60, 74), (60, top - 7.5)],
          color=GOOD, width=1.8, text="β")

    # floating (nacelle fore-aft velocity) feedback gain into pitch Σ
    _block(fig, 104, low + 34, 30, 12, "Floating feedback", "K_F = 0.05  (nacelle vel.)",
           color=ACCENT, fill=_FILL_SOFT, tsize=10)
    _poly(fig, [(104, low + 40), (104, mid + 3.2)], color=ACCENT, width=1.5, dash="dot")
    # nacelle velocity source arrow into the KF block
    _poly(fig, [(140, top - 7.5), (140, low + 34), (119, low + 34)],
          color=ACCENT, width=1.3, dash="dot", text="nacelle ẋ")
    return fig


# --- 3. 6-DOF Cummins coupled hydro -----------------------------------------
def hydro_block() -> go.Figure:
    """6-DOF Cummins time-domain equation as a coupled-system block diagram."""
    plat = VOLTURNUS_S
    Ts, Tp = plat.surge_natural_period_s, plat.pitch_natural_period_s

    xr, yr = (0.0, 184.0), (0.0, 122.0)
    fig = _sheet("PLATFORM HYDRODYNAMICS — 6-DOF CUMMINS MODEL",
                 "potential-flow BEM + radiation memory + mooring + wave/thrust forcing",
                 "HAL-BLK-03", xr, yr, height=610,
                 notes=[
                     f"Validated natural periods: surge {Ts:.0f} s · heave 19.2 s · "
                     f"pitch {Tp:.0f} s · yaw 85.3 s.",
                     "q = [surge, sway, heave, roll, pitch, yaw]ᵀ.",
                     "K(t) = radiation-retardation kernel from B(ω).",
                 ], notes_corner="bottom-left")

    # equation banner
    by0, by1 = 100.0, 114.0
    fig.add_shape(type="path",
                  path=_round_rect_path(10, by0, 174, by1, 3.0),
                  line=dict(color=INK, width=1.8), fillcolor=_FILL_SOFT)
    fig.add_annotation(
        x=92, y=(by0 + by1) / 2,
        text=("<b>(M + A<sub>∞</sub>) q̈ + ∫₀ᵗ K(t−τ) q̇(τ) dτ + C q + "
              "F<sub>mooring</sub> = X(ω) η(t) + F<sub>thrust</sub></b>"),
        showarrow=False, font=dict(size=13, color=INK))

    # central solver block
    cxc, cyc = 118.0, 56.0
    _block(fig, cxc, cyc, 40, 22, "6-DOF EOM solve",
           "time-domain integrator  →  q(t)", color=ACCENT, tsize=12, lever=True)

    # left input stack feeding the solver
    inputs = [
        (86, "Potential-flow BEM", "Capytaine (offline): A(ω), B(ω), X(ω)", WARN),
        (72, "Radiation kernel K(t)", "convolution memory ← B(ω)", ACCENT),
        (58, "Hydrostatic stiffness C", "analytic (waterplane + CoB)", ACCENT),
        (44, "Linearised mooring", "catenary  F_mooring", GOOD),
        (30, "Morison drag", "viscous  ~ q̇|q̇|", GOOD),
    ]
    lbx = 40.0
    for cy, title, sub, col in inputs:
        _block(fig, lbx, cy, 46, 10.5, title, sub, color=col, tsize=10, ssize=8)
        _arrow(fig, lbx + 23, cy, cxc - 20, cyc + (cy - 58) * 0.28,
               color=BP_LINE, width=1.3)

    # forcing inputs from the top
    _block(fig, 150, 86, 40, 12, "Wave spectrum", "JONSWAP → η(t)", color=WARN,
           fill=_FILL_SOFT)
    _arrow(fig, 150, 80, cxc + 8, cyc + 11, color=BP_LINE, width=1.5,
           text="X(ω)·η", tpos="above")
    _block(fig, 150, 30, 40, 12, "Rotor thrust", "aero-servo coupling", color=WARN,
           fill=_FILL_SOFT)
    _arrow(fig, 150, 36, cxc + 8, cyc - 11, color=BP_LINE, width=1.5,
           text="F_thrust", tpos="below")

    # output: platform motion → dynamic cable
    _block(fig, 118, 18, 40, 12, "Dynamic cable", "hang-off curvature & tension",
           color=GOOD)
    _arrow(fig, cxc, cyc - 11, 118, 24, color=ACCENT, width=2.0, text="q(t)",
           tpos="above")

    # feedback: platform velocity → radiation memory + Morison drag
    _poly(fig, [(cxc + 20, cyc + 6), (166, cyc + 6), (166, 72), (lbx + 23 + 42, 72),
                (lbx + 23, 72)], color=ACCENT, width=1.3, dash="dot",
          text="q̇ (velocity feedback)")
    _poly(fig, [(166, cyc + 6), (166, 30), (lbx + 23, 30)], color=ACCENT,
          width=1.3, dash="dot")
    return fig


# --- 4. end-to-end analysis pipeline ---------------------------------------
def pipeline_flow() -> go.Figure:
    """End-to-end fatigue-analysis pipeline with the Phase-1 honesty gate."""
    xr, yr = (0.0, 192.0), (0.0, 126.0)
    fig = _sheet("ANALYSIS PIPELINE — METOCEAN → FATIGUE COST",
                 "DLC matrix · coupled sim · cable dynamics · rainflow · S-N · Miner",
                 "HAL-BLK-04", xr, yr, height=620,
                 notes=[
                     "Phase-1 honesty gate routes each case to Stage A or Stage B.",
                     "DFF = design fatigue factor;  DAF = dynamic amplification factor.",
                 ], notes_corner="bottom-left")

    # legend (block / diamond / data)
    lx, ly = 20.0, 108.0
    fig.add_shape(type="path", path=_round_rect_path(lx, ly, lx + 8, ly + 4, 0.8),
                  line=dict(color=ACCENT, width=1.6), fillcolor=_FILL)
    fig.add_annotation(x=lx + 10, y=ly + 2, text="process", showarrow=False,
                       xanchor="left", font=dict(size=8, color=INK))
    dcx = lx + 30
    fig.add_shape(type="path",
                  path=f"M {dcx},{ly + 4} L {dcx + 4},{ly + 2} L {dcx},{ly} "
                       f"L {dcx - 4},{ly + 2} Z",
                  line=dict(color=_LEVER, width=1.6), fillcolor=_FILL_SOFT)
    fig.add_annotation(x=dcx + 6, y=ly + 2, text="decision", showarrow=False,
                       xanchor="left", font=dict(size=8, color=INK))
    fig.add_annotation(x=dcx + 26, y=ly + 2,
                       text="—— data flow   ····· fidelity branch", showarrow=False,
                       xanchor="left", font=dict(size=8, color=GREY))

    bw, bh = 34.0, 15.0

    # Stage 0: metocean & DLC matrix
    _block(fig, 30, 86, bw, bh + 4, "Metocean & DLC matrix",
           "Weibull wind × sea state × heading × seeds", color=ACCENT)
    # Phase-1 honesty gate diamond
    _diamond(fig, 82, 86, 30, 26, "Phase-1 gate", "screening pass?")
    _arrow(fig, 47, 86, 67, 86, color=BP_LINE, text="cases", tpos="above")

    # Stage A (reduced-order live) / Stage B (high-fidelity ingested)
    _block(fig, 138, 104, bw + 6, bh, "Stage A · coupled sim",
           "grid SFR + ROSCO + 6-DOF BEM hydro", color=WARN)
    _block(fig, 138, 68, bw + 6, bh, "Stage B · ingested",
           "high-fidelity OpenFAST / OrcaFlex decks", color=WARN, fill=_FILL_SOFT)
    _poly(fig, [(82, 99), (82, 104), (114, 104)], color=BP_LINE, width=1.4,
          text="A: reduced-order")
    _poly(fig, [(82, 73), (82, 68), (114, 68)], color=BP_LINE, width=1.4,
          dash="dot", text="B: high-fidelity")

    # merge into cable dynamics
    mx = 138.0
    _block(fig, mx, 40, bw + 6, bh, "Coupled cable dynamics",
           "quasi-static family + DAF · dynamic FE + VIV", color=GOOD)
    _poly(fig, [(mx, 96.5), (mx, 47.5)], color=BP_LINE, width=1.4)
    _poly(fig, [(mx, 60.5), (mx, 47.5)], color=BP_LINE, width=1.4, dash="dot")

    # bottom processing chain (right → left)
    row = 16.0
    chain = [
        (162, "Equivalent-armour\nstress", "σ from κ, T", GOOD),
        (126, "Rainflow", "cycle counting", ACCENT),
        (92, "Mean-stress", "Goodman / Gerber", ACCENT),
        (58, "S-N curve", "DNV-RP-C203", ACCENT),
        (24, "Miner + DFF", "Σ nᵢ/Nᵢ · DFF", INK),
    ]
    cxs = [c[0] for c in chain]
    for cx, title, sub, col in chain:
        _block(fig, cx, row, 30, 14, title.replace("\n", "<br>"), sub, color=col,
               tsize=10, ssize=8)
    # cable dynamics → first chain box (down then left)
    _poly(fig, [(mx, 32.5), (mx, 23), (162, 23)], color=BP_LINE, width=1.4,
          text="stress ranges")
    for a, b in zip(cxs[:-1], cxs[1:]):
        _arrow(fig, a - 15, row, b + 15, row, color=BP_LINE, width=1.5)

    # final: annualised life & COST
    _block(fig, 24, 44, 30, 15, "Life & fatigue COST",
           "annualised €/yr", color=_LEVER, lever=True)
    _arrow(fig, 24, row + 7, 24, 36.5, color=_LEVER, width=2.0, text="D → life",
           tpos="above")
    return fig
