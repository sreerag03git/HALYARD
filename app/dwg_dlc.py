"""Fatigue / DLC story drawings — matrix + metocean scatter (§10, §4).

Two engineering figures built from the load-case distributions (no full ``run_dlc``
simulation): the IEC DLC-1.2-style fatigue matrix with its aggregation flow, and the
Hs-Tp metocean scatter diagram overlaid with the Weibull wind occurrence and the
wind-correlated sea-state locus. Blueprint drafting furniture from ``app.dwg_kit`` for
the matrix; house ``theme.plotly_layout`` for the scatter. Headless (no streamlit).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from app import dwg_kit as kit
from app import theme
from app.theme import ACCENT, GREY, INK, WARN, GOOD, GRID_LINE
from app.dwg_kit import BP_LINE, BP_INK
from analysis.dlc import sea_state_for_wind, wind_weibull_weights
from models.metocean import SCATTER

_G = 9.81  # gravity [m/s^2]

# DLC-1.2 fatigue matrix axes (mirror analysis.dlc defaults).
_WINDS = (8.0, 11.0, 14.0, 17.0)
_HEADINGS = (0.0, 45.0, 90.0)
_SN_LOGN_STD = 0.20  # S-N scatter (log10 N) propagated to the life band


def _lerp_rgb(c0: str, c1: str, t: float) -> str:
    """Linear blend between two #rrggbb colours; t in [0,1] -> 'rgb(r,g,b)'."""
    t = min(1.0, max(0.0, t))
    a = tuple(int(c0[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
    r = tuple(int(round(a[j] + (b[j] - a[j]) * t)) for j in range(3))
    return f"rgb({r[0]},{r[1]},{r[2]})"


def dlc_matrix_diagram() -> go.Figure:
    """IEC DLC-1.2-style fatigue matrix + aggregation flow (blueprint schematic).

    Grid of load cases (wind bins x wave headings), each cell shaded by its combined
    occurrence weight and annotated with the wind-correlated sea state and weight;
    alongside, the matrix -> multi-seed sims -> rainflow+Miner -> Weibull-weighted
    annual sum -> design-life band aggregation flow.
    """
    winds = np.asarray(_WINDS, float)
    heads = np.asarray(_HEADINGS, float)
    wgt_wind = wind_weibull_weights(winds)          # normalised occurrence per wind bin
    head_wgt = np.ones(len(heads)) / len(heads)     # uniform heading occurrence
    weights = np.outer(wgt_wind, head_wgt)          # [nwind, nhead] combined weight
    wmin, wmax = float(weights.min()), float(weights.max())

    xr, yr = (0.0, 162.0), (0.0, 100.0)
    fig = go.Figure()
    kit.blueprint_axes(fig, xr, yr, height=640, equal=False)
    kit.frame(fig, xr, yr, zones=True)

    # ---- fatigue matrix (left) --------------------------------------------
    mx0, mx1 = 24.0, 86.0
    my0, my1 = 30.0, 76.0
    nw, nh = len(winds), len(heads)
    cw = (mx1 - mx0) / nw
    rh = (my1 - my0) / nh

    fig.add_annotation(x=(mx0 + mx1) / 2, y=my1 + 15, text="<b>DLC-1.2 FATIGUE MATRIX</b>",
                       showarrow=False, font=dict(size=12, color=BP_INK))
    fig.add_annotation(x=(mx0 + mx1) / 2, y=my1 + 10,
                       text="load cases = wind bin × wave heading · cell = combined occurrence",
                       showarrow=False, font=dict(size=8.5, color=BP_LINE))
    # axis captions
    fig.add_annotation(x=(mx0 + mx1) / 2, y=my1 + 5.2, text="wind speed  V  (m/s)  →  Weibull A=10, k=2",
                       showarrow=False, font=dict(size=9, color=BP_INK))
    fig.add_annotation(x=mx0 - 12.5, y=(my0 + my1) / 2, text="wave heading (deg)", textangle=-90,
                       showarrow=False, font=dict(size=9, color=BP_INK))

    for iw, V in enumerate(winds):
        Hs, Tp = sea_state_for_wind(float(V))
        cx = mx0 + (iw + 0.5) * cw
        # column header (wind bin)
        fig.add_annotation(x=cx, y=my1 + 1.6, text=f"<b>{V:.0f}</b>", showarrow=False,
                           font=dict(size=10, color=BP_INK))
        for ih, hd in enumerate(heads):
            w = float(weights[iw, ih])
            t = (w - wmin) / (wmax - wmin) if wmax > wmin else 0.5
            fill = _lerp_rgb("#EAF1F7", ACCENT, t)
            x0 = mx0 + iw * cw
            y0 = my1 - (ih + 1) * rh          # heading 0 at the top row
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x0 + cw, y1=y0 + rh,
                          line=dict(color=BP_LINE, width=0.9), fillcolor=fill)
            tcol = "#FFFFFF" if t > 0.5 else BP_INK
            ycen = y0 + rh / 2
            fig.add_annotation(x=x0 + cw / 2, y=ycen + 2.6,
                               text=f"Hs {Hs:.2f} m", showarrow=False,
                               font=dict(size=8, color=tcol))
            fig.add_annotation(x=x0 + cw / 2, y=ycen + 0.2,
                               text=f"Tp {Tp:.1f} s", showarrow=False,
                               font=dict(size=8, color=tcol))
            fig.add_annotation(x=x0 + cw / 2, y=ycen - 2.6,
                               text=f"<b>w={w:.4f}</b>", showarrow=False,
                               font=dict(size=8, color=tcol))
            if iw == 0:                        # row header (heading)
                fig.add_annotation(x=mx0 - 2.5, y=ycen, text=f"<b>{hd:.0f}°</b>",
                                   showarrow=False, xanchor="right",
                                   font=dict(size=10, color=BP_INK))

    # matrix weight legend (graduated bar under the grid)
    lx0, lx1 = mx0, mx0 + 34.0
    ly = my0 - 8.0
    nseg = 6
    for i in range(nseg):
        t = (i + 0.5) / nseg
        sx = lx0 + i * (lx1 - lx0) / nseg
        fig.add_shape(type="rect", x0=sx, y0=ly, x1=sx + (lx1 - lx0) / nseg, y1=ly + 2.6,
                      line=dict(color=BP_LINE, width=0.5), fillcolor=_lerp_rgb("#EAF1F7", ACCENT, t))
    fig.add_annotation(x=lx0, y=ly - 1.0, text=f"{wmin:.4f}", showarrow=False, yanchor="top",
                       xanchor="left", font=dict(size=7.5, color=BP_LINE))
    fig.add_annotation(x=lx1, y=ly - 1.0, text=f"{wmax:.4f}", showarrow=False, yanchor="top",
                       xanchor="right", font=dict(size=7.5, color=BP_LINE))
    fig.add_annotation(x=(lx0 + lx1) / 2, y=ly + 4.4, text="combined occurrence weight",
                       showarrow=False, font=dict(size=8, color=BP_INK))
    fig.add_annotation(x=lx1 + 12.0, y=ly + 1.3,
                       text=f"Σw = {weights.sum():.3f}   ·   {nw}×{nh} = {nw * nh} bins",
                       showarrow=False, font=dict(size=8, color=BP_LINE))

    # ---- aggregation flow (right column) ----------------------------------
    fx0, fx1 = 100.0, 150.0
    steps = [
        ("DLC-1.2 matrix", f"{nw}×{nh} = {nw * nh} env. bins", ACCENT),
        ("multi-seed coupled sims", "2 wave seeds × support on/off", ACCENT),
        ("rainflow + Miner per bin", "hang-off armour stress → D", WARN),
        ("Weibull-weighted annual sum", "Σ wᵢ · Dᵢ  →  1/(D·DFF)", GOOD),
        ("design life", "mean + 5–95% band", INK),
    ]
    fig.add_annotation(x=(fx0 + fx1) / 2, y=my1 + 15, text="<b>AGGREGATION</b>",
                       showarrow=False, font=dict(size=12, color=BP_INK))
    n = len(steps)
    bh = 9.5
    top = my1 + 8.0
    gap = (top - 14.0 - bh) / (n - 1)          # even spacing down to y≈14
    for i, (title, sub, col) in enumerate(steps):
        by1 = top - i * gap
        by0 = by1 - bh
        fig.add_shape(type="rect", x0=fx0, y0=by0, x1=fx1, y1=by1,
                      line=dict(color=col, width=1.8), fillcolor="rgba(255,255,255,0.96)")
        fig.add_annotation(x=(fx0 + fx1) / 2, y=by0 + bh * 0.66, text=f"<b>{title}</b>",
                           showarrow=False, font=dict(size=9.5, color=INK))
        fig.add_annotation(x=(fx0 + fx1) / 2, y=by0 + bh * 0.28, text=sub,
                           showarrow=False, font=dict(size=8, color=BP_LINE))
        if i < n - 1:
            ny1 = top - (i + 1) * gap
            fig.add_annotation(x=(fx0 + fx1) / 2, y=ny1, ax=(fx0 + fx1) / 2, ay=by0,
                               xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               text="", arrowhead=3, arrowsize=1.1, arrowwidth=1.4,
                               arrowcolor=GREY)
    # leader from the matrix into the flow
    fig.add_annotation(x=fx0, y=top - bh / 2, ax=mx1, ay=(my0 + my1) / 2, xref="x", yref="y",
                       axref="x", ayref="y", showarrow=True, text="", arrowhead=3,
                       arrowsize=1.1, arrowwidth=1.2, arrowcolor=GREY)

    kit.notes_block(fig, xr, yr, [
        "Fatigue matrix per IEC 61400-3 DLC-1.2 (normal power production).",
        "Wind occurrence: Weibull A=10 m/s, k=2, normalised over the bins.",
        "Wave heading occurrence uniform (1/3 per bin); Hs, Tp wind-correlated.",
        "Sea state per bin: Hs = 0.25 + 0.14·V, Tp = 4.5 + 0.45·V.",
        f"5–95% band combines seed scatter and S-N log-N scatter σ_logN = {_SN_LOGN_STD:.2f}.",
    ], corner="bottom-left", title="NOTES")

    kit.revision_table(fig, xr, yr, [("A", "issued — computed distributions")])
    kit.title_block(fig, xr, yr, "IEC DLC-1.2-style fatigue matrix",
                    subtitle="wind × heading load cases → annualized hang-off life",
                    dwg_no="HAL-DLC-01", scale="NTS", rev="A",
                    extra="from wind_weibull_weights + sea_state_for_wind")
    return fig


def metocean_scatter_diagram() -> go.Figure:
    """Hs-Tp metocean scatter: occurrence bubbles, iso-steepness lines, wind locus.

    The 7 representative scatter bins as area-proportional bubbles coloured by
    occurrence, with deep-water iso-steepness reference lines, the wind-correlated
    sea-state locus (Hs, Tp)(V), and the Weibull wind occurrence overlaid on a
    secondary axis (mapped through Tp = 4.5 + 0.45·V).
    """
    Tp = np.array([s.Tp_s for s in SCATTER])
    Hs = np.array([s.Hs_m for s in SCATTER])
    occ = np.array([s.occurrence for s in SCATTER])
    occ_max = float(occ.max())

    fig = go.Figure()

    # --- deep-water iso-steepness reference curves: Hs = (1/N)·g·Tp²/(2π) ----
    tp_line = np.linspace(4.0, 16.0, 120)
    for N in (15, 20, 30):
        hs_line = (1.0 / N) * _G * tp_line ** 2 / (2 * np.pi)
        m = hs_line <= 8.2
        fig.add_trace(go.Scatter(x=tp_line[m], y=hs_line[m], mode="lines",
                                 line=dict(color=GREY, width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False))
        # label near the top of each curve inside the frame
        xi = float(tp_line[m][-1])
        yi = float(hs_line[m][-1])
        fig.add_annotation(x=xi, y=yi, text=f"s = 1/{N}", showarrow=False, xshift=-4,
                           yshift=6, font=dict(size=9, color=GREY),
                           bgcolor="rgba(255,255,255,0.75)")

    # --- wind-correlated sea-state locus (Hs, Tp)(V) ------------------------
    Vloc = np.linspace(3.0, 25.0, 60)
    Tp_loc = 4.5 + 0.45 * Vloc
    Hs_loc = np.maximum(0.5, 0.25 + 0.14 * Vloc)
    fig.add_trace(go.Scatter(x=Tp_loc, y=Hs_loc, mode="lines",
                             line=dict(color=WARN, width=2.2),
                             name="wind-correlated sea state (V)", hoverinfo="skip"))
    for V in (5, 10, 15, 20, 25):
        tpv, hsv = 4.5 + 0.45 * V, max(0.5, 0.25 + 0.14 * V)
        fig.add_trace(go.Scatter(x=[tpv], y=[hsv], mode="markers", marker=dict(color=WARN, size=5),
                                 hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=tpv, y=hsv, text=f"{V} m/s", showarrow=False, yshift=-11,
                           font=dict(size=8, color=WARN))

    # --- Weibull wind occurrence on a secondary axis (mapped via Tp) ---------
    Vgrid = np.linspace(3.0, 25.0, 240)
    wgrid = wind_weibull_weights(Vgrid)          # normalised over the grid
    Tp_w = 4.5 + 0.45 * Vgrid
    fig.add_trace(go.Scatter(x=Tp_w, y=wgrid, mode="lines", yaxis="y2",
                             line=dict(color=ACCENT, width=1.4, dash="dash"),
                             fill="tozeroy", fillcolor="rgba(58,90,120,0.08)",
                             name="Weibull wind occurrence (A=10, k=2)", hoverinfo="skip"))

    # --- scatter bins as area-proportional occurrence bubbles ---------------
    fig.add_trace(go.Scatter(
        x=Tp, y=Hs, mode="markers+text",
        marker=dict(size=occ, sizemode="area", sizeref=2.0 * occ_max / (58.0 ** 2), sizemin=6,
                    color=occ, colorscale=[[0.0, "#EAF1F7"], [1.0, ACCENT]],
                    line=dict(color=BP_INK, width=1.0),
                    colorbar=dict(title=dict(text="occurrence", side="right"),
                                  thickness=12, len=0.5, x=1.005, y=0.5,
                                  tickformat=".0%")),
        text=[f"{o * 100:.0f}%" for o in occ], textposition="middle center",
        textfont=dict(size=9, color=BP_INK),
        name="scatter bins", hoverinfo="skip"))
    # occurrence % as a leader label above each bubble too (small bins get a clear tag)
    for tpi, hsi, oi in zip(Tp, Hs, occ):
        fig.add_annotation(x=tpi, y=hsi, text=f"{oi * 100:.0f}%", showarrow=False,
                           yshift=int(10 + 46 * (oi / occ_max) ** 0.5),
                           font=dict(size=8, color=INK))

    fig = theme.plotly_layout(fig, "Metocean scatter — Hs vs Tp (representative)",
                              height=440, xtitle="Tp — peak period (s)",
                              ytitle="Hs — significant wave height (m)", legend=True)
    fig.update_xaxes(range=[4.0, 16.0])
    fig.update_yaxes(range=[0.0, 8.0])
    fig.update_layout(
        yaxis2=dict(title=dict(text="wind occurrence (Weibull)", font=dict(color=ACCENT)),
                    overlaying="y", side="right", showgrid=False, rangemode="tozero",
                    tickfont=dict(color=ACCENT, size=10), position=1.0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(r=70),
    )
    return fig
