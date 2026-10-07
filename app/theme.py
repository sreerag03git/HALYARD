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

# Responsive rules shared by the (max-width) media query and the explicit "compact /
# mobile layout" toggle. They stack every st.columns row, let the horizontal section
# nav wrap, and tighten padding/metrics so the app is usable on a phone. Kept as one
# block (no selectors specific to a single screen width) so both callers stay in sync.
# Streamlit >=1.4 exposes columns as [data-testid="stColumn"]; the legacy "column"
# testid is kept as a fallback so this is version-robust across the pinned 1.54 / local.
_RESPONSIVE_RULES = """
.block-container, [data-testid="stMainBlockContainer"] {
    padding-left: 0.9rem !important; padding-right: 0.9rem !important;
}
h1 { font-size: 1.35rem !important; }
h2 { font-size: 1.15rem !important; }
h3 { font-size: 1.02rem !important; }
[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.5rem !important; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important;
}
div[role="radiogroup"] { flex-wrap: wrap !important; gap: 0.25rem 0.7rem !important; }
div[role="radiogroup"] label { font-size: 0.82rem !important; }
div[data-testid="stMetric"] { padding: 8px 10px !important; }
div[data-testid="stMetricValue"] { font-size: 1.15rem !important; }
"""


def inject_css():
    import streamlit as st
    base = f"""
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
        /* Automatic phone / small-tablet layout (no device detection needed). */
        @media (max-width: 820px) {{{_RESPONSIVE_RULES}}}
        </style>
        """
    # st.html injects raw HTML straight into the DOM (NOT an iframe, unlike
    # components.html), so global <style> rules apply. Crucially it bypasses the markdown
    # parser, which otherwise re-enters markdown mode on the column-0 CSS lines inside the
    # @media block and renders them as literal text.
    st.html(base)


def inject_mobile_css():
    """Force the compact layout at ANY viewport width — driven by the sidebar toggle,
    so a desktop user can opt into the phone layout (e.g. a narrow window or screenshot)."""
    import streamlit as st
    st.html(f"<style>{_RESPONSIVE_RULES}</style>")


def boot_splash() -> str:
    """A neat, on-brand loading overlay for the first paint of the session.

    Pure CSS: st.markdown strips <script>, so the overlay dismisses itself with a CSS
    animation (fades out, then becomes hidden + non-interactive via fill-mode: forwards)
    rather than any JavaScript. Rendered once per session (gated on session_state), so it
    covers the cold-start boot but never flickers on ordinary reruns. Honours
    prefers-reduced-motion. Colours/type come from the design system above."""
    return f"""
    <style>
    @keyframes halyardSpin {{ to {{ transform: rotate(360deg); }} }}
    @keyframes halyardLoad {{
        0%   {{ transform: translateX(-130%); }}
        60%  {{ transform: translateX(300%); }}
        100% {{ transform: translateX(300%); }}
    }}
    @keyframes halyardSplashOut {{
        0%, 60% {{ opacity: 1; visibility: visible; }}
        100%    {{ opacity: 0; visibility: hidden; pointer-events: none; }}
    }}
    .halyard-splash {{
        position: fixed; inset: 0; z-index: 100000;
        display: flex; align-items: center; justify-content: center;
        background: radial-gradient(1100px 560px at 50% 20%, #FFFFFF 0%, {PAPER} 58%, #F0F0EB 100%);
        animation: halyardSplashOut 2.4s ease-in forwards;
    }}
    .halyard-splash .inner {{ text-align: center; transform: translateY(-4px); }}
    .halyard-splash .ring {{
        width: 52px; height: 52px; margin: 0 auto 22px;
        border: 3px solid {GRID_LINE}; border-top-color: {ACCENT}; border-radius: 50%;
        animation: halyardSpin 0.9s linear infinite;
    }}
    .halyard-splash .word {{
        font-family: {FONT_STACK}; font-weight: 700; letter-spacing: 0.24em;
        font-size: 1.7rem; color: {INK}; padding-left: 0.24em;
    }}
    .halyard-splash .sub {{ color: #55554F; font-size: 0.82rem; margin-top: 7px; letter-spacing: 0.02em; }}
    .halyard-splash .bar {{
        width: 188px; height: 3px; margin: 22px auto 0; border-radius: 3px;
        background: {GRID_LINE}; overflow: hidden;
    }}
    .halyard-splash .bar > span {{
        display: block; height: 100%; width: 38%; border-radius: 3px; background: {ACCENT};
        animation: halyardLoad 1.4s ease-in-out infinite;
    }}
    .halyard-splash .tag {{ color: {GREY}; font-size: 0.72rem; margin-top: 15px; letter-spacing: 0.03em; }}
    @media (prefers-reduced-motion: reduce) {{
        .halyard-splash .ring, .halyard-splash .bar > span {{ animation: none; }}
        .halyard-splash {{ animation-duration: 1.0s; }}
    }}
    </style>
    <div class="halyard-splash">
      <div class="inner">
        <div class="ring"></div>
        <div class="word">HALYARD</div>
        <div class="sub">Cable-fatigue cost of grid-frequency support</div>
        <div class="bar"><span></span></div>
        <div class="tag">IEA-15&nbsp;MW &middot; VolturnUS-S &middot; initialising physics engine</div>
      </div>
    </div>
    """


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
