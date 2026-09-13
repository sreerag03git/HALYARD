"""Fatigue engineering diagrams — DNV-RP-C203 S-N, Miner accumulation, rainflow spectrum (§5.7).

Three code-style fatigue figures built from the real fatigue results (never invented
geometry): the DNV-RP-C203 bilinear design curve with the counted cycles overlaid, the
Palmgren-Miner damage accumulation by cable region, and the rainflow stress-range
spectrum with a cumulative-exceedance overlay. Every public function returns a
``go.Figure`` styled through :func:`app.theme.plotly_layout`; no streamlit, no matplotlib.

Consumes the fatigue objects produced upstream (``FatigueResult`` and ``CaseResult``);
solves nothing itself. Damage counting is always Miner on the COMBINED stress signal —
these figures visualise that single-signal accounting, never a wave/control split.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from app import theme
from app.theme import ACCENT, GOOD, GREY, GRID_LINE, INK, PAPER, SURFACE, WARN
from physics.cable import REGIONS
from physics.fatigue import rainflow_histogram

# Ink for the design curve / near-black annotation furniture.
_CURVE = INK
_BOX_BG = "rgba(255,255,255,0.90)"


def _info_box(fig: go.Figure, x: float, y: float, lines: list[str],
              xanchor: str = "left", yanchor: str = "top",
              color: str = INK, border: str = GRID_LINE) -> None:
    """A bordered annotation cell (a mini title-block) in paper coordinates."""
    fig.add_annotation(x=x, y=y, xref="paper", yref="paper", text="<br>".join(lines),
                       showarrow=False, xanchor=xanchor, yanchor=yanchor, align="left",
                       font=dict(size=9, color=color), bgcolor=_BOX_BG,
                       bordercolor=border, borderwidth=1, borderpad=6)


# ---------------------------------------------------------------------------
def sn_diagram(fat, sn) -> go.Figure:
    """DNV-RP-C203-style bilinear S-N design curve with the counted cycles overlaid.

    Draws the design curve S(N) from ``sn.cycles_to_failure`` over a logspace stress
    sweep, labels the slope(s) m1/m2 and the knee, overlays the counted equivalent
    ranges as markers, and annotates the Design Fatigue Factor and the Miner sum on the
    combined signal. Log-log axes with decade grid.
    """
    S = np.logspace(0.0, 3.2, 240)
    N = np.asarray(sn.cycles_to_failure(S), float)
    finite = np.isfinite(N) & (N > 0)
    S_c, N_c = S[finite], N[finite]

    fig = go.Figure()

    # Design curve (thick near-black code line).
    fig.add_trace(go.Scatter(x=N_c, y=S_c, mode="lines", name="design curve (DNV-RP-C203)",
                             line=dict(color=_CURVE, width=2.6)))

    # Knee between the two slopes (curve constants are MPa-based).
    m1 = float(getattr(sn, "m1", 3.0))
    m2 = float(getattr(sn, "m2", m1))
    log_a1 = float(getattr(sn, "log_a1", 11.764))
    N_knee = float(getattr(sn, "N_knee", 1.0e6))
    S_knee = 10.0 ** ((log_a1 - np.log10(N_knee)) / m1)
    two_slope = abs(m2 - m1) > 1e-9
    if two_slope and np.isfinite(S_knee):
        fig.add_vline(x=N_knee, line=dict(color=GREY, width=1, dash="dot"))
        fig.add_trace(go.Scatter(x=[N_knee], y=[S_knee], mode="markers",
                                 marker=dict(color=_CURVE, size=8, symbol="circle-open",
                                             line=dict(width=1.6, color=_CURVE)),
                                 name="knee", showlegend=False))
        fig.add_annotation(x=np.log10(N_knee), y=np.log10(max(S_knee, 1e-6)),
                           text=f"knee  N={N_knee:.0e}", showarrow=True, arrowhead=3,
                           arrowsize=1.0, arrowwidth=1.0, arrowcolor=GREY, ax=-38, ay=-26,
                           font=dict(size=9, color=GREY), bgcolor=_BOX_BG,
                           bordercolor=GRID_LINE, borderwidth=1)

    # Slope callouts placed along each segment.
    def _slope_label(N_at: float, m: float, tag: str) -> None:
        if not (np.isfinite(N_at) and N_at > 0):
            return
        seg = np.where(S >= S_knee, m1, m2) if two_slope else np.full_like(S, m1)
        # evaluate the curve stress at N_at by nearest node
        idx = int(np.argmin(np.abs(N_c - N_at))) if len(N_c) else 0
        if not len(N_c):
            return
        fig.add_annotation(x=np.log10(N_c[idx]), y=np.log10(max(S_c[idx], 1e-6)),
                           text=f"<b>{tag}</b>  m = {m:g}", showarrow=False, xshift=30,
                           yshift=16, font=dict(size=10, color=_CURVE), bgcolor=_BOX_BG,
                           bordercolor=GRID_LINE, borderwidth=1)

    _slope_label(N_knee / 50.0, m1, "S1")
    if two_slope:
        _slope_label(N_knee * 60.0, m2, "S2")

    # Overlay counted cycles (mean-corrected equivalent ranges) on the curve.
    S_eq = np.asarray(getattr(fat, "S_eq_MPa", np.array([])), float)
    N_i = np.asarray(getattr(fat, "N_i", np.array([])), float)
    counts = np.asarray(getattr(fat, "counts", np.array([])), float)
    if S_eq.size and N_i.size:
        ok = np.isfinite(N_i) & (N_i > 0) & (S_eq > 0)
        if np.any(ok):
            csz = counts[ok] if counts.size == ok.size else np.ones(int(np.sum(ok)))
            cmax = float(np.max(csz)) if csz.size else 1.0
            sizes = 5.0 + 9.0 * np.sqrt(np.clip(csz, 0, None) / (cmax or 1.0))
            fig.add_trace(go.Scatter(
                x=N_i[ok], y=S_eq[ok], mode="markers", name="counted cycles (S_eq)",
                marker=dict(color=ACCENT, size=sizes, opacity=0.6,
                            line=dict(width=0.5, color=SURFACE)),
                hovertemplate="N=%{x:.2e}<br>S_eq=%{y:.1f} MPa<extra></extra>"))

    fig.update_xaxes(type="log", dtick=1, minor=dict(showgrid=True, gridcolor=GRID_LINE,
                                                     griddash="dot"))
    fig.update_yaxes(type="log", dtick=1, minor=dict(showgrid=True, gridcolor=GRID_LINE,
                                                     griddash="dot"))

    # Design Fatigue Factor + Miner-sum assessment note.
    DFF = float(getattr(getattr(fat, "params", None), "DFF", 3.0))
    dmg = float(getattr(fat, "damage", 0.0))
    ncyc = float(getattr(fat, "n_cycles", 0.0))
    _info_box(fig, 0.015, 0.03, [
        "<b>MINER SUM (combined signal)</b>",
        f"D = &#931; n&#7522;/N&#7522; = {dmg:.2e} / window",
        f"counted cycles &#931;n&#7522; = {ncyc:.0f}",
        f"Design Fatigue Factor DFF = {DFF:g}",
        "allowable life = 1 / (D&#775; &#183; DFF)",
    ], xanchor="left", yanchor="bottom")

    return theme.plotly_layout(
        fig, "S-N (DNV-RP-C203 style)", height=420,
        xtitle="cycles to failure  N", ytitle="equivalent stress range  S_eq (MPa)")


# ---------------------------------------------------------------------------
def damage_accumulation(case) -> go.Figure:
    """Palmgren-Miner damage accumulation by cable region — a fatigue-assessment summary.

    Horizontal bars of per-region Miner damage on a log axis (hang-off concentration),
    each labelled with its damage, share of total and counted cycles, plus the D=1
    failure-line concept and a summary caption.
    """
    regions = list(REGIONS)
    dmg = np.array([float(case.fatigue[r].damage) for r in regions], float)
    ncyc = np.array([float(getattr(case.fatigue[r], "n_cycles", 0.0)) for r in regions], float)
    total = float(np.sum(dmg))
    colors = [theme.REGION_COLORS[r] for r in regions]
    labels = [r.replace("_", "-") for r in regions]

    # Floor for the log axis so zero-damage regions still render a stub + a valid range.
    pos = dmg[dmg > 0]
    floor = (float(np.min(pos)) * 1e-2) if pos.size else 1e-30
    bar_x = np.where(dmg > 0, dmg, floor)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bar_x, y=labels, orientation="h", marker=dict(color=colors, line=dict(color=INK,
                                                                                width=0.6)),
        width=0.55, showlegend=False, hoverinfo="skip"))

    # Per-region callouts: damage value, % of total, counted cycles.
    for lab, d, n in zip(labels, dmg, ncyc):
        share = (100.0 * d / total) if total > 0 else 0.0
        fig.add_annotation(x=np.log10(max(d, floor)), y=lab,
                           text=f"  D = {d:.2e}  ({share:.1f}% &#183; n={n:.0f})",
                           showarrow=False, xanchor="left", yanchor="middle",
                           font=dict(size=10, color=INK), bgcolor=_BOX_BG,
                           bordercolor=GRID_LINE, borderwidth=1)

    # D = 1 failure reference (Miner criterion) and the window-total reference.
    fig.add_vline(x=1.0, line=dict(color=WARN, width=1.6, dash="dash"))
    fig.add_annotation(x=0.0, y=1.0, xref="x", yref="paper", text="<b>D = 1  (failure)</b>",
                       showarrow=False, xanchor="center", yanchor="bottom", yshift=2,
                       font=dict(size=10, color=WARN))
    if total > 0:
        fig.add_vline(x=total, line=dict(color=GREY, width=1, dash="dot"))
        fig.add_annotation(x=np.log10(total), y=0.0, xref="x", yref="paper",
                           text=f"&#931; window D = {total:.2e}", showarrow=False,
                           xanchor="center", yanchor="bottom", yshift=2,
                           font=dict(size=9, color=GREY))

    dom = labels[int(np.argmax(dmg))] if dmg.size else "-"
    dom_share = (100.0 * float(np.max(dmg)) / total) if total > 0 else 0.0
    _info_box(fig, 0.985, 0.05, [
        "<b>PALMGREN-MINER ACCUMULATION</b>",
        f"dominant region: {dom} ({dom_share:.0f}% of window D)",
        "failure when cumulative D &#8805; 1 over service life",
        "damage per window &#8594; annualise &#8594; life = 1/(D&#775;&#183;DFF)",
    ], xanchor="right", yanchor="bottom", color=INK)

    xhi = max(1.0, total) * 10.0
    fig.update_xaxes(type="log", range=[np.log10(floor), np.log10(xhi)], dtick=2)
    return theme.plotly_layout(
        fig, "Palmgren-Miner damage accumulation by region", height=380,
        xtitle="Miner damage per window  D  (log)", ytitle="", legend=False)


# ---------------------------------------------------------------------------
def rainflow_matrix(fat) -> go.Figure:
    """Rainflow stress-range spectrum (hang-off) with a cumulative-exceedance overlay.

    Bars of counted cycles per range bin (from :func:`physics.fatigue.rainflow_histogram`)
    with the range-exceedance staircase (cycles with range >= r) on a secondary axis.
    Guards empty counting windows.
    """
    centres, counts = rainflow_histogram(fat)
    fig = go.Figure()

    if not len(centres) or not len(counts):
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                           text="no counted cycles in window", showarrow=False,
                           font=dict(size=12, color=GREY))
        return theme.plotly_layout(
            fig, "Rainflow stress-range spectrum (hang-off)", height=380,
            xtitle="stress range  S (MPa)", ytitle="cycle count", legend=False)

    centres = np.asarray(centres, float)
    counts = np.asarray(counts, float)

    # Range spectrum bars.
    fig.add_trace(go.Bar(x=centres, y=counts, name="cycles per range bin",
                         marker=dict(color=ACCENT, line=dict(color=INK, width=0.4)),
                         opacity=0.85))

    # Cumulative exceedance: cycles with range >= each bin centre (staircase).
    exceed = np.cumsum(counts[::-1])[::-1]
    fig.add_trace(go.Scatter(x=centres, y=exceed, name="exceedance (range &#8805; S)",
                             mode="lines", line=dict(color=WARN, width=2, shape="hv"),
                             yaxis="y2"))

    # Engineering callouts: total cycles, max range, mean stress span.
    ncyc = float(getattr(fat, "n_cycles", float(np.sum(counts))))
    means = np.asarray(getattr(fat, "means_MPa", np.array([])), float)
    lines = ["<b>RAINFLOW COUNT (ASTM E1049)</b>",
             f"&#931; cycles = {ncyc:.0f}",
             f"max range = {float(np.max(centres)):.1f} MPa"]
    if means.size:
        lines.append(f"mean stress {float(np.min(means)):.0f}&#8211;{float(np.max(means)):.0f} MPa")
    _info_box(fig, 0.985, 0.95, lines, xanchor="right", yanchor="top")

    fig = theme.plotly_layout(
        fig, "Rainflow stress-range spectrum (hang-off)", height=400,
        xtitle="stress range  S (MPa)", ytitle="cycle count per bin", legend=True)
    fig.update_layout(
        yaxis2=dict(title="cumulative exceedance (cycles)", overlaying="y", side="right",
                    showgrid=False, type="log", tickfont=dict(size=11, color=WARN),
                    title_font=dict(size=11, color=WARN)),
        legend=dict(x=0.5, y=0.98, xanchor="center", yanchor="top", orientation="h"))
    return fig
