"""Reusable Plotly drafting toolkit (§8) — the shared drawing kit.

Blueprint-style primitives every 2-D drawing module composes: invisible true-scale
axes, a double drawing border with zone grid, a corner title block, revision and
notes blocks, dimension lines (horizontal/vertical/radial/angular), leaders,
hatching, centrelines, section markers, a graduated scale bar, a north arrow and a
waterline. Pure functions that mutate the passed go.Figure in DATA coordinates;
no global state, no streamlit.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import plotly.graph_objects as go

from app.theme import INK

# Blueprint palette ---------------------------------------------------------
BP_BG = "#F4F7FA"        # cold blueprint paper
BP_LINE = "#37506B"      # dimension / border rule
BP_INK = "#12202E"       # near-black annotation ink
STEEL = "#7C8B99"        # steelwork mid tone
STEEL_D = "#5E6B78"      # steelwork shade
SEABED = "#D9CDB0"       # seabed fill
SEABED_LINE = "#B7A57F"  # seabed hatch / rule
WATER = "#AEC6DE"        # water tint
BUOY = "#E0A03A"         # buoyancy module amber
CABLE_BLACK = "#20242A"  # cable jacket

# Title-block geometry (fractions of the drawing extents) -------------------
_TB_W = 0.30
_TB_H = 0.14
_TB_MARGIN = 0.028


# --- internal helpers ------------------------------------------------------
def _span(xr: Sequence[float], yr: Sequence[float]) -> Tuple[float, float]:
    return float(xr[1] - xr[0]), float(yr[1] - yr[0])


def _arrowhead(fig: go.Figure, x: float, y: float, ax_px: float, ay_px: float,
               color: str, width: float = 1.2) -> None:
    """One arrowhead landing on (x,y); tail is (ax_px, ay_px) pixels away."""
    fig.add_annotation(x=x, y=y, ax=ax_px, ay=ay_px, xref="x", yref="y",
                       axref="pixel", ayref="pixel", text="", showarrow=True,
                       arrowhead=3, arrowsize=1.1, arrowwidth=width, arrowcolor=color)


def _tb_box(xr: Sequence[float], yr: Sequence[float]) -> Tuple[float, float, float, float]:
    dx, dy = _span(xr, yr)
    bx1 = xr[1] - _TB_MARGIN * dx
    bx0 = bx1 - _TB_W * dx
    by0 = yr[0] + _TB_MARGIN * dy
    by1 = by0 + _TB_H * dy
    return bx0, by0, bx1, by1


# --- axes & frame ----------------------------------------------------------
def blueprint_axes(fig: go.Figure, xr: Sequence[float], yr: Sequence[float],
                   height: int = 620, equal: bool = True) -> go.Figure:
    """Invisible true-scale axes on a cold blueprint ground; returns the figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BP_BG,
        margin=dict(l=8, r=8, t=8, b=8), height=height, showlegend=False,
        font=dict(color=INK),
    )
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False,
                     range=[xr[0], xr[1]], constrain="domain")
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False,
                     range=[yr[0], yr[1]], constrain="domain")
    if equal:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # invisible anchor trace pins the data range so shapes/annotations render true-scale
    fig.add_trace(go.Scatter(x=[xr[0], xr[1]], y=[yr[0], yr[1]], mode="markers",
                             marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    return fig


def frame(fig: go.Figure, xr: Sequence[float], yr: Sequence[float],
          zones: bool = True) -> None:
    """Double drawing border inset from the extents, with an optional zone grid."""
    dx, dy = _span(xr, yr)
    po = 0.018                         # outer inset fraction
    pi = 0.030                         # inner inset fraction
    ox0, ox1 = xr[0] + po * dx, xr[1] - po * dx
    oy0, oy1 = yr[0] + po * dy, yr[1] - po * dy
    ix0, ix1 = xr[0] + pi * dx, xr[1] - pi * dy * 0 - pi * dx
    iy0, iy1 = yr[0] + pi * dy, yr[1] - pi * dy
    for (x0, y0, x1, y1, w) in ((ox0, oy0, ox1, oy1, 1.6), (ix0, iy0, ix1, iy1, 0.9)):
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=BP_LINE, width=w), fillcolor="rgba(0,0,0,0)")
    if not zones:
        return
    ncols, nrows = 8, 5
    letters = "ABCDEFGHIJ"
    for j in range(ncols):
        xc = ix0 + (j + 0.5) * (ix1 - ix0) / ncols
        for yy in (0.5 * (oy1 + iy1), 0.5 * (oy0 + iy0)):
            fig.add_annotation(x=xc, y=yy, text=str(j + 1), showarrow=False,
                               font=dict(size=8, color=BP_LINE))
    for i in range(nrows):
        yc = iy1 - (i + 0.5) * (iy1 - iy0) / nrows
        for xx in (0.5 * (ox0 + ix0), 0.5 * (ox1 + ix1)):
            fig.add_annotation(x=xx, y=yc, text=letters[i], showarrow=False,
                               font=dict(size=8, color=BP_LINE))
    # zone tick marks along the inner border
    for j in range(1, ncols):
        xg = ix0 + j * (ix1 - ix0) / ncols
        fig.add_shape(type="line", x0=xg, y0=iy1, x1=xg, y1=oy1,
                      line=dict(color=BP_LINE, width=0.6))
        fig.add_shape(type="line", x0=xg, y0=iy0, x1=xg, y1=oy0,
                      line=dict(color=BP_LINE, width=0.6))
    for i in range(1, nrows):
        yg = iy0 + i * (iy1 - iy0) / nrows
        fig.add_shape(type="line", x0=ix0, y0=yg, x1=ox0, y1=yg,
                      line=dict(color=BP_LINE, width=0.6))
        fig.add_shape(type="line", x0=ix1, y0=yg, x1=ox1, y1=yg,
                      line=dict(color=BP_LINE, width=0.6))


def title_block(fig: go.Figure, xr: Sequence[float], yr: Sequence[float], title: str,
                subtitle: str = "", dwg_no: str = "", scale: str = "NTS", rev: str = "A",
                extra: str = "computed geometry") -> None:
    """Multi-cell title block in the bottom-right corner."""
    bx0, by0, bx1, by1 = _tb_box(xr, yr)
    h = by1 - by0
    # outer box
    fig.add_shape(type="rect", x0=bx0, y0=by0, x1=bx1, y1=by1,
                  line=dict(color=BP_LINE, width=1.4), fillcolor="#FFFFFF")
    # row rules at 30% (org band) and 60% (title band)
    r1 = by1 - 0.30 * h
    r2 = by1 - 0.62 * h
    for yy in (r1, r2):
        fig.add_shape(type="line", x0=bx0, y0=yy, x1=bx1, y1=yy,
                      line=dict(color=BP_LINE, width=0.8))
    pad = 0.012 * (xr[1] - xr[0])
    # organisation band
    fig.add_annotation(x=bx0 + pad, y=(by1 + r1) / 2, text="<b>HALYARD</b>",
                       showarrow=False, xanchor="left", yanchor="middle",
                       font=dict(size=11, color=BP_INK))
    fig.add_annotation(x=bx1 - pad, y=(by1 + r1) / 2,
                       text="floating-wind cable fatigue", showarrow=False,
                       xanchor="right", yanchor="middle", font=dict(size=8, color=BP_LINE))
    # title + subtitle band
    fig.add_annotation(x=bx0 + pad, y=(r1 + r2) / 2 + 0.06 * h, text=f"<b>{title}</b>",
                       showarrow=False, xanchor="left", yanchor="middle",
                       font=dict(size=10, color=INK))
    if subtitle:
        fig.add_annotation(x=bx0 + pad, y=(r1 + r2) / 2 - 0.07 * h, text=subtitle,
                           showarrow=False, xanchor="left", yanchor="middle",
                           font=dict(size=8, color=BP_LINE))
    # bottom band: three cells DWG NO | SCALE | REV
    c1 = bx0 + 0.52 * (bx1 - bx0)
    c2 = bx0 + 0.78 * (bx1 - bx0)
    for xx in (c1, c2):
        fig.add_shape(type="line", x0=xx, y0=by0, x1=xx, y1=r2,
                      line=dict(color=BP_LINE, width=0.8))
    cells = ((bx0, c1, "DWG NO", dwg_no or "—"), (c1, c2, "SCALE", scale),
             (c2, bx1, "REV", rev))
    for a, b, lab, val in cells:
        fig.add_annotation(x=a + 0.5 * pad, y=r2 - 0.06 * h, text=lab, showarrow=False,
                           xanchor="left", yanchor="top", font=dict(size=7, color=BP_LINE))
        fig.add_annotation(x=(a + b) / 2, y=(by0 + r2) / 2 - 0.02 * h, text=f"<b>{val}</b>",
                           showarrow=False, xanchor="center", yanchor="middle",
                           font=dict(size=9, color=BP_INK))
    if extra:
        fig.add_annotation(x=bx0 + pad, y=by0 + 0.06 * h, text=extra, showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=7, color=BP_LINE))


def revision_table(fig: go.Figure, xr: Sequence[float], yr: Sequence[float],
                   rows: Sequence[Tuple[str, str]]) -> None:
    """Small revision table sitting just above the title block."""
    bx0, _, bx1, by1 = _tb_box(xr, yr)
    dx, dy = _span(xr, yr)
    rows = list(rows) or [("A", "issued")]
    rh = 0.026 * dy
    gap = 0.012 * dy
    tbot = by1 + gap
    ttop = tbot + rh * (len(rows) + 1)
    fig.add_shape(type="rect", x0=bx0, y0=tbot, x1=bx1, y1=ttop,
                  line=dict(color=BP_LINE, width=1.0), fillcolor="#FFFFFF")
    cx = bx0 + 0.18 * (bx1 - bx0)
    fig.add_shape(type="line", x0=cx, y0=tbot, x1=cx, y1=ttop,
                  line=dict(color=BP_LINE, width=0.7))
    pad = 0.010 * dx
    # header
    hy = ttop - 0.5 * rh
    fig.add_shape(type="line", x0=bx0, y0=ttop - rh, x1=bx1, y1=ttop - rh,
                  line=dict(color=BP_LINE, width=0.7))
    fig.add_annotation(x=bx0 + pad, y=hy, text="<b>REV</b>", showarrow=False,
                       xanchor="left", yanchor="middle", font=dict(size=7, color=BP_LINE))
    fig.add_annotation(x=cx + pad, y=hy, text="<b>DESCRIPTION</b>", showarrow=False,
                       xanchor="left", yanchor="middle", font=dict(size=7, color=BP_LINE))
    for k, (rv, desc) in enumerate(rows):
        yy = ttop - rh - (k + 0.5) * rh
        fig.add_annotation(x=bx0 + pad, y=yy, text=str(rv), showarrow=False,
                           xanchor="left", yanchor="middle", font=dict(size=8, color=BP_INK))
        fig.add_annotation(x=cx + pad, y=yy, text=str(desc), showarrow=False,
                           xanchor="left", yanchor="middle", font=dict(size=8, color=INK))
        if k < len(rows) - 1:
            fig.add_shape(type="line", x0=bx0, y0=yy - 0.5 * rh, x1=bx1, y1=yy - 0.5 * rh,
                          line=dict(color=BP_LINE, width=0.4))


def notes_block(fig: go.Figure, xr: Sequence[float], yr: Sequence[float],
                notes: Sequence[str], corner: str = "top-left", title: str = "NOTES") -> None:
    """Numbered notes list anchored to one corner of the drawing."""
    dx, dy = _span(xr, yr)
    m = 0.045
    if corner.endswith("left"):
        ax, anchor = xr[0] + m * dx, "left"
    else:
        ax, anchor = xr[1] - m * dx, "right"
    top = corner.startswith("top")
    ay = (yr[1] - m * dy) if top else (yr[0] + m * dy + (len(notes) + 1) * 0.030 * dy)
    fig.add_annotation(x=ax, y=ay, text=f"<b>{title}</b>", showarrow=False,
                       xanchor=anchor, yanchor="top", font=dict(size=10, color=BP_INK))
    lh = 0.030 * dy
    for i, note in enumerate(notes):
        fig.add_annotation(x=ax, y=ay - (i + 1) * lh, text=f"{i + 1}.  {note}",
                           showarrow=False, xanchor=anchor, yanchor="top",
                           align="left" if anchor == "left" else "right",
                           font=dict(size=9, color=INK))


# --- dimensions ------------------------------------------------------------
def dim_h(fig: go.Figure, x1: float, x2: float, y: float, text: str, ext: float = 0.0,
          color: str = BP_LINE) -> None:
    """Horizontal dimension at height y with extension lines, double arrows, label."""
    if ext:
        for xe in (x1, x2):
            fig.add_shape(type="line", x0=xe, y0=y - ext, x1=xe, y1=y,
                          line=dict(color=color, width=0.7))
    fig.add_shape(type="line", x0=x1, y0=y, x1=x2, y1=y, line=dict(color=color, width=0.9))
    _arrowhead(fig, x1, y, 16, 0, color)
    _arrowhead(fig, x2, y, -16, 0, color)
    fig.add_annotation(x=(x1 + x2) / 2, y=y, text=text, showarrow=False, yshift=8,
                       font=dict(size=9, color=BP_INK), bgcolor="rgba(244,247,250,0.85)")


def dim_v(fig: go.Figure, y1: float, y2: float, x: float, text: str, ext: float = 0.0,
          color: str = BP_LINE) -> None:
    """Vertical dimension at abscissa x with a rotated label."""
    if ext:
        for ye in (y1, y2):
            fig.add_shape(type="line", x0=x - ext, y0=ye, x1=x, y1=ye,
                          line=dict(color=color, width=0.7))
    fig.add_shape(type="line", x0=x, y0=y1, x1=x, y1=y2, line=dict(color=color, width=0.9))
    _arrowhead(fig, x, y1, 0, -16, color)
    _arrowhead(fig, x, y2, 0, 16, color)
    fig.add_annotation(x=x, y=(y1 + y2) / 2, text=text, showarrow=False, xshift=-9,
                       textangle=-90, font=dict(size=9, color=BP_INK),
                       bgcolor="rgba(244,247,250,0.85)")


def dim_radial(fig: go.Figure, cx: float, cy: float, r: float, ang_deg: float, text: str,
               diameter: bool = False) -> None:
    """Radial (or diameter) dimension arrow from centre outward at ang_deg."""
    a = math.radians(ang_deg)
    ux, uy = math.cos(a), math.sin(a)
    ex, ey = cx + r * ux, cy + r * uy
    if diameter:
        sx, sy = cx - r * ux, cy - r * uy
        fig.add_shape(type="line", x0=sx, y0=sy, x1=ex, y1=ey, line=dict(color=BP_LINE, width=0.9))
        _arrowhead(fig, sx, sy, 14 * ux, 14 * uy, BP_LINE)
        _arrowhead(fig, ex, ey, -14 * ux, -14 * uy, BP_LINE)
        lab = text if text.upper().startswith(("DIA", "⌀")) else f"⌀ {text}"
        lx, ly = cx, cy
    else:
        fig.add_shape(type="line", x0=cx, y0=cy, x1=ex, y1=ey, line=dict(color=BP_LINE, width=0.9))
        _arrowhead(fig, ex, ey, -14 * ux, -14 * uy, BP_LINE)
        lab = text if text.upper().startswith("R") else f"R {text}"
        lx, ly = cx + 0.55 * r * ux, cy + 0.55 * r * uy
    fig.add_annotation(x=lx, y=ly, text=lab, showarrow=False, font=dict(size=9, color=BP_INK),
                       bgcolor="rgba(244,247,250,0.85)")


def dim_angular(fig: go.Figure, cx: float, cy: float, r: float, a0_deg: float, a1_deg: float,
                text: str) -> None:
    """Angular dimension arc between two rays with a centred label."""
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    for a in (a0, a1):
        fig.add_shape(type="line", x0=cx, y0=cy, x1=cx + r * math.cos(a),
                      y1=cy + r * math.sin(a), line=dict(color=BP_LINE, width=0.7))
    n = max(2, int(abs(a1_deg - a0_deg) / 4) + 1)
    pts = [(cx + r * math.cos(a0 + (a1 - a0) * t / n), cy + r * math.sin(a0 + (a1 - a0) * t / n))
           for t in range(n + 1)]
    path = "M " + " L ".join(f"{px},{py}" for px, py in pts)
    fig.add_shape(type="path", path=path, line=dict(color=BP_LINE, width=0.9))
    _arrowhead(fig, pts[0][0], pts[0][1],
               14 * math.sin(a0), -14 * math.cos(a0), BP_LINE)
    _arrowhead(fig, pts[-1][0], pts[-1][1],
               -14 * math.sin(a1), 14 * math.cos(a1), BP_LINE)
    am = 0.5 * (a0 + a1)
    fig.add_annotation(x=cx + 0.62 * r * math.cos(am), y=cy + 0.62 * r * math.sin(am),
                       text=text, showarrow=False, font=dict(size=9, color=BP_INK),
                       bgcolor="rgba(244,247,250,0.85)")


def leader(fig: go.Figure, x: float, y: float, text: str, dx: float = 0.0, dy: float = 0.0,
           color: str = BP_INK, size: int = 10) -> None:
    """Leader line with arrowhead landing on (x,y); text at the elbow (x+dx, y+dy)."""
    if dx == 0.0 and dy == 0.0:                     # pixel-offset elbow fallback
        fig.add_annotation(x=x, y=y, ax=42, ay=-34, xref="x", yref="y",
                           axref="pixel", ayref="pixel", text=text, showarrow=True,
                           arrowhead=3, arrowsize=1.1, arrowwidth=1.0, arrowcolor=color,
                           font=dict(size=size, color=color), bgcolor="rgba(244,247,250,0.85)")
        return
    fig.add_annotation(x=x, y=y, ax=x + dx, ay=y + dy, xref="x", yref="y", axref="x",
                       ayref="y", text=text, showarrow=True, arrowhead=3, arrowsize=1.1,
                       arrowwidth=1.0, arrowcolor=color, font=dict(size=size, color=color),
                       bgcolor="rgba(244,247,250,0.85)")


# --- fills, lines, markers -------------------------------------------------
def hatch_band(fig: go.Figure, x0: float, x1: float, y0: float, y1: float,
               angle_deg: float = 45.0, spacing: Optional[float] = None,
               color: str = SEABED_LINE, width: float = 0.6,
               fill: Optional[str] = None) -> None:
    """Fill a rectangle with evenly spaced hatch lines (optional solid fill under)."""
    xlo, xhi = min(x0, x1), max(x0, x1)
    ylo, yhi = min(y0, y1), max(y0, y1)
    if fill is not None:
        fig.add_shape(type="rect", x0=xlo, y0=ylo, x1=xhi, y1=yhi, line=dict(width=0),
                      fillcolor=fill, layer="below")
    if spacing is None:
        spacing = max(xhi - xlo, yhi - ylo) / 18.0
    if spacing <= 0:
        return
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)               # line direction
    nx, ny = -math.sin(a), math.cos(a)              # unit normal
    corners = ((xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi))
    cs = [px * nx + py * ny for px, py in corners]
    c = math.floor(min(cs) / spacing) * spacing
    cmax = max(cs)
    while c <= cmax:
        # line: p(t) = c*(nx,ny) + t*(dx,dy); clip to rectangle
        tlo, thi = -1e18, 1e18
        ok = True
        for lo, hi, oc, od in ((xlo, xhi, c * nx, dx), (ylo, yhi, c * ny, dy)):
            if abs(od) < 1e-12:
                if oc < lo or oc > hi:
                    ok = False
                    break
            else:
                ta, tb = (lo - oc) / od, (hi - oc) / od
                if ta > tb:
                    ta, tb = tb, ta
                tlo, thi = max(tlo, ta), min(thi, tb)
        if ok and thi > tlo:
            ax, ay = c * nx + tlo * dx, c * ny + tlo * dy
            bx, by = c * nx + thi * dx, c * ny + thi * dy
            fig.add_shape(type="line", x0=ax, y0=ay, x1=bx, y1=by,
                          line=dict(color=color, width=width), layer="below")
        c += spacing


def centerline(fig: go.Figure, x0: float, y0: float, x1: float, y1: float,
               color: str = BP_LINE) -> None:
    """Dash-dot centreline."""
    fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                  line=dict(color=color, width=0.8, dash="dashdot"))


def section_marker(fig: go.Figure, x: float, y: float, label: str, angle_deg: float = 0.0) -> None:
    """Section cut flag with a bold label and a direction arrow."""
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    fig.add_annotation(x=x, y=y, text=f"<b>{label}</b>", showarrow=False,
                       font=dict(size=11, color="#FFFFFF"), bgcolor=BP_INK,
                       bordercolor=BP_INK, borderpad=3)
    # direction arrow off the flag
    fig.add_annotation(x=x + 22 * ux, y=y + 22 * uy, ax=x, ay=y, xref="x", yref="y",
                       axref="x", ayref="y", text="", showarrow=True,
                       arrowhead=3, arrowsize=1.2, arrowwidth=1.4, arrowcolor=BP_INK)


def scale_bar(fig: go.Figure, x: float, y: float, length_m: float, n: int = 4,
              unit: str = "m", color: str = BP_INK) -> None:
    """Graduated (alternating filled/empty) scale bar of true length length_m."""
    n = max(1, int(n))
    seg = length_m / n
    h = seg / 3.0
    for i in range(n):
        sx = x + i * seg
        fig.add_shape(type="rect", x0=sx, y0=y, x1=sx + seg, y1=y + h,
                      line=dict(color=color, width=0.8),
                      fillcolor=color if i % 2 == 0 else "#FFFFFF")
    for i in range(n + 1):
        tx = x + i * seg
        fig.add_annotation(x=tx, y=y, text=f"{i * seg:g}", showarrow=False, yshift=-8,
                           font=dict(size=8, color=color))
    fig.add_annotation(x=x + length_m, y=y + h, text=unit, showarrow=False, xshift=10,
                       yanchor="middle", font=dict(size=8, color=color))


def north_arrow(fig: go.Figure, x: float, y: float, size: float, label: str = "N") -> None:
    """North arrow (filled triangle + label) for plan views."""
    w = 0.32 * size
    path = f"M {x},{y + size} L {x - w},{y} L {x},{y + 0.28 * size} L {x + w},{y} Z"
    fig.add_shape(type="path", path=path, line=dict(color=BP_INK, width=0.8), fillcolor=BP_INK)
    fig.add_annotation(x=x, y=y + size, text=f"<b>{label}</b>", showarrow=False, yshift=8,
                       font=dict(size=10, color=BP_INK))


def waterline(fig: go.Figure, x0: float, x1: float, z: float = 0.0, label: str = "MSL") -> None:
    """Dashed water-surface line with a small MSL flag."""
    fig.add_shape(type="line", x0=x0, y0=z, x1=x1, y1=z,
                  line=dict(color=WATER, width=1.4, dash="dash"))
    fig.add_annotation(x=x0, y=z, text=label, showarrow=False, xanchor="left", yshift=7,
                       xshift=2, font=dict(size=8, color="#4E6B86"),
                       bgcolor="rgba(244,247,250,0.85)")
