"""HALYARD design system — light, premium, restrained (§8).

One accent (muted steel/slate blue), warm greys, near-black ink on near-paper ground.
Tabular figures for numbers, generous whitespace, no emojis, no decorative graphics.
The three controllers keep the SAME colours across every chart (colour-blind-safe).
"""
from __future__ import annotations

# Palette -------------------------------------------------------------------
INK = "#1A1A1A"
PAPER = "#FAFAF8"
SURFACE = "#FFFFFF"
ACCENT = "#3A5A78"          # muted steel/slate blue — primary actions & key data lines
GREY = "#8A8A86"
GRID_LINE = "#E6E6E1"
GOOD = "#2E6B4F"            # muted green (PASS)
WARN = "#B4632C"            # muted amber/terracotta

# Fixed, colour-blind-safe controller colours — identical across ALL charts (§8).
C_NO_SUPPORT = "#9AA3AB"    # grey  — wave-only baseline
C_INCUMBENT = "#E69F00"     # amber — cited incumbent recovery
C_HALYARD = ACCENT          # steel blue — HALYARD shaped recovery
CONTROLLER_COLORS = {"no_support": C_NO_SUPPORT, "incumbent": C_INCUMBENT,
                     "halyard": C_HALYARD}
CONTROLLER_LABELS = {"no_support": "No support (wave only)",
                     "incumbent": "Incumbent recovery",
                     "halyard": "HALYARD shaped recovery"}

# Region colours for the cable.
REGION_COLORS = {"hang_off": ACCENT, "sag": "#7FA8C9", "hog": "#C98F3A",
                 "touchdown": "#9AA3AB"}

FONT_STACK = ("Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
              "Helvetica, Arial, sans-serif")


def inject_css():
    import streamlit as st
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: {FONT_STACK}; color: {INK}; }}
        .stApp {{ background: {PAPER}; }}
        /* Tabular figures everywhere numbers matter */
        .stMetric, .stMarkdown, table, .dataframe {{ font-variant-numeric: tabular-nums; }}
        h1, h2, h3 {{ font-weight: 600; letter-spacing: -0.01em; color: {INK}; }}
        h1 {{ font-size: 1.7rem; }}
        /* Restrained metric cards */
        div[data-testid="stMetric"] {{
            background: {SURFACE}; border: 1px solid {GRID_LINE};
            border-radius: 8px; padding: 12px 16px;
        }}
        div[data-testid="stMetricValue"] {{ font-size: 1.4rem; }}
        /* Tabs: quiet, underlined active */
        button[data-baseweb="tab"] {{ font-weight: 500; }}
        /* Sidebar surface */
        section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {GRID_LINE}; }}
        .halyard-tag {{
            display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
            text-transform: uppercase; padding: 3px 9px; border-radius: 5px; margin-right: 6px;
        }}
        .tag-reduced {{ background: #EEF2F6; color: {ACCENT}; border: 1px solid #D6E0EA; }}
        .tag-highfi {{ background: #F3EEE6; color: {WARN}; border: 1px solid #E6D8C6; }}
        .whatshows {{ color: #55554F; font-size: 0.86rem; border-left: 3px solid {GRID_LINE};
            padding: 2px 0 2px 12px; margin: 4px 0 14px 0; }}
        .verdict-pass {{ color: {GOOD}; font-weight: 700; }}
        .verdict-fail {{ color: {WARN}; font-weight: 700; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def fidelity_tag(kind: str = "reduced") -> str:
    if kind == "highfi":
        return '<span class="halyard-tag tag-highfi">high-fidelity (ingested)</span>'
    return '<span class="halyard-tag tag-reduced">reduced-order (live)</span>'


def what_this_shows(text: str) -> str:
    return f'<div class="whatshows">{text}</div>'


def plotly_layout(fig, title: str = "", height: int = 360, xtitle: str = "",
                  ytitle: str = "", legend: bool = True):
    """Apply the consistent engineering-figure style to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
        font=dict(family=FONT_STACK, size=12, color=INK),
        margin=dict(l=60, r=20, t=40 if title else 16, b=48),
        height=height, showlegend=legend,
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=GRID_LINE, borderwidth=1,
                    font=dict(size=11)),
    )
    fig.update_xaxes(title_text=xtitle, gridcolor=GRID_LINE, zerolinecolor=GRID_LINE,
                     linecolor=GRID_LINE, ticks="outside", tickcolor=GRID_LINE)
    fig.update_yaxes(title_text=ytitle, gridcolor=GRID_LINE, zerolinecolor=GRID_LINE,
                     linecolor=GRID_LINE, ticks="outside", tickcolor=GRID_LINE)
    return fig
