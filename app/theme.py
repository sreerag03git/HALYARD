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


# The loading page. Injected via a components.html iframe whose script writes the overlay
# straight into the PARENT document (window.parent) — i.e. OUTSIDE Streamlit's managed element
# tree. A plain st.html/st.markdown overlay is a tracked element: Streamlit's immediate
# post-load rerun does not re-emit it, so element-diffing deletes it and it never shows. An
# overlay appended to document.body survives every rerun. It dismisses itself with a
# self-contained CSS animation (plus a parent-window setTimeout safety net), so it does not
# depend on the (short-lived) iframe surviving. A parent-window flag shows it once per page
# load (every open/reload) but not on reruns. The content is typographic (no illustration): an
# "H" monogram mark (two masts + a sagging halyard/cable line that draws itself in, with a light
# pulse travelling it), the HALYARD wordmark, the project's one-line purpose, and two numbered
# points — "the claim" / "the test", framing the Phase-1 honesty gate. Palette as above.
#
# NOTE: on Streamlit Community Cloud a cold/asleep container first shows Streamlit's OWN
# "waking up" / loading screen, served before any app code runs — this overlay covers the
# phase after that, while the script builds the page, and cannot replace that platform screen.
_SPLASH_HTML = """
<script>
(function () {
  try {
    var win = window.parent, doc = win.document;
    if (!doc || doc.getElementById('halyard-splash')) return;      // overlay already in the page
    // Guard on a parent-window flag, NOT sessionStorage: it lives only as long as the page's
    // JS context, so the splash shows once per page LOAD (every open/reload) but is skipped on
    // Streamlit reruns (slider changes), which keep the same context.
    if (win.__halyardSplashShown) return;
    win.__halyardSplashShown = true;

    var css = doc.createElement('style');
    css.id = 'halyard-splash-css';
    css.textContent =
      '@keyframes hlRise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}' +
      '@keyframes hlLine{from{transform:scaleX(0)}to{transform:scaleX(1)}}' +
      '@keyframes hlSweep{0%{left:-35%}100%{left:100%}}' +
      '@keyframes hlDraw{from{stroke-dashoffset:100}to{stroke-dashoffset:0}}' +
      '@keyframes hlTravel{from{stroke-dashoffset:178}to{stroke-dashoffset:14}}' +
      '@keyframes hlOut{0%,85%{opacity:1;visibility:visible}100%{opacity:0;visibility:hidden}}' +
      '#halyard-splash{position:fixed;inset:0;z-index:2147483647;display:flex;flex-direction:column;' +
      'align-items:center;justify-content:center;padding:24px;box-sizing:border-box;background:#FAFAF8;' +
      "pointer-events:none;font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;" +
      'animation:hlOut 3.8s ease-in forwards}' +
      '#halyard-splash .hl-mark{width:50px;height:50px;opacity:0;animation:hlRise .6s ease-out .05s forwards}' +
      '#halyard-splash .hl-mark svg{display:block;width:50px;height:50px;overflow:visible}' +
      '#halyard-splash .hl-mast,#halyard-splash .hl-rope{stroke:#3A5A78;fill:none;stroke-linecap:round;' +
      'stroke-dasharray:100;stroke-dashoffset:0}' +
      '#halyard-splash .hl-mast{stroke-width:5;animation:hlDraw .55s ease-out both}' +
      '#halyard-splash .hl-mast-l{animation-delay:.12s}#halyard-splash .hl-mast-r{animation-delay:.24s}' +
      '#halyard-splash .hl-rope{stroke-width:3.4;animation:hlDraw .5s ease-out .54s both}' +
      '#halyard-splash .hl-rope-glow{stroke:#86ADD6;fill:none;stroke-width:3.4;stroke-linecap:round;' +
      'stroke-dasharray:14 150;stroke-dashoffset:178;animation:hlTravel 1.7s linear 1.1s infinite}' +
      '#halyard-splash .hl-node{fill:#3A5A78}' +
      '#halyard-splash .hl-w{opacity:0;margin-top:26px;font-weight:700;letter-spacing:.14em;font-size:1.7rem;' +
      'color:#1A1A1A;padding-left:.14em;animation:hlRise .6s ease-out .18s forwards}' +
      '#halyard-splash .hl-rule{width:46px;height:2px;margin-top:14px;border-radius:2px;background:#3A5A78;' +
      'transform:scaleX(0);animation:hlLine .5s ease-out .42s forwards}' +
      '#halyard-splash .hl-s{opacity:0;max-width:440px;margin-top:16px;color:#55554F;font-size:.9rem;' +
      'line-height:1.5;text-align:center;animation:hlRise .6s ease-out .54s forwards}' +
      '#halyard-splash .hl-points{display:flex;flex-wrap:wrap;justify-content:center;gap:28px;' +
      'margin-top:30px;max-width:540px}' +
      '#halyard-splash .hl-pt{flex:1 1 210px;max-width:234px;min-width:186px;text-align:left;opacity:0;' +
      'animation:hlRise .6s ease-out forwards}' +
      '#halyard-splash .hl-pt.p1{animation-delay:.72s}#halyard-splash .hl-pt.p2{animation-delay:.9s}' +
      '#halyard-splash .hl-n{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.72rem;' +
      'letter-spacing:.14em;color:#3A5A78}' +
      '#halyard-splash .hl-h{margin-top:7px;font-weight:600;font-size:.96rem;color:#1A1A1A}' +
      '#halyard-splash .hl-b{margin-top:5px;color:#8A8A86;font-size:.8rem;line-height:1.5}' +
      '#halyard-splash .hl-bar{opacity:0;position:relative;width:min(460px,78vw);height:3px;margin-top:34px;' +
      'border-radius:3px;background:#E9E9E4;overflow:hidden;animation:hlRise .5s ease-out 1.05s forwards}' +
      '#halyard-splash .hl-bar>i{position:absolute;top:0;left:-35%;height:100%;width:35%;border-radius:3px;' +
      'background:linear-gradient(90deg,rgba(58,90,120,0),#3A5A78 50%,rgba(58,90,120,0));' +
      'animation:hlSweep 1.25s cubic-bezier(.65,.05,.36,1) 1.25s infinite}' +
      '@media (prefers-reduced-motion:reduce){#halyard-splash *{opacity:1!important;transform:none!important;' +
      'animation:none!important}#halyard-splash .hl-mast,#halyard-splash .hl-rope{stroke-dashoffset:0!important}' +
      '#halyard-splash .hl-rope-glow{opacity:0!important}#halyard-splash .hl-bar>i{left:0;width:100%}}';
    doc.head.appendChild(css);

    var ov = doc.createElement('div');
    ov.id = 'halyard-splash';
    ov.innerHTML =
      '<div class="hl-mark" aria-hidden="true">' +
        '<svg viewBox="0 0 52 52" xmlns="http://www.w3.org/2000/svg" fill="none">' +
          '<line class="hl-mast hl-mast-l" x1="14" y1="43" x2="14" y2="9" pathLength="100"/>' +
          '<line class="hl-mast hl-mast-r" x1="38" y1="43" x2="38" y2="9" pathLength="100"/>' +
          '<path class="hl-rope" d="M14 26 Q26 34 38 26" pathLength="100"/>' +
          '<path class="hl-rope-glow" d="M14 26 Q26 34 38 26" pathLength="100"/>' +
          '<circle class="hl-node" cx="14" cy="26" r="2.4"/>' +
          '<circle class="hl-node" cx="38" cy="26" r="2.4"/>' +
        '</svg>' +
      '</div>' +
      '<div class="hl-w">HALYARD</div>' +
      '<div class="hl-rule"></div>' +
      '<div class="hl-s">A physics test bench for the cable-fatigue cost of grid-frequency ' +
      'support — on a floating offshore wind turbine.</div>' +
      '<div class="hl-points">' +
        '<div class="hl-pt p1"><div class="hl-n">01</div><div class="hl-h">The claim</div>' +
        '<div class="hl-b">Grid-frequency support flexes the dynamic export cable and adds ' +
        'measurable hang-off fatigue.</div></div>' +
        '<div class="hl-pt p2"><div class="hl-n">02</div><div class="hl-h">The test</div>' +
        '<div class="hl-b">Quantify that cost against a committed, pre-registered honesty ' +
        'gate — or fail it plainly.</div></div>' +
      '</div>' +
      '<div class="hl-bar"><i></i></div>';
    doc.body.appendChild(ov);

    // Safety net on the PARENT window, so removal survives this iframe being torn down.
    win.setTimeout(function () { if (ov && ov.parentNode) ov.parentNode.removeChild(ov); }, 4200);
  } catch (e) {}
})();
</script>
"""


def boot_splash():
    """Render the loading page (see _SPLASH_HTML). Call on EVERY run (unconditional): the
    component iframe must be re-emitted each run so Streamlit's element-diffing never tears it
    down before its script runs. The parent-window guard inside makes the overlay appear only
    once per page load, so reruns don't re-show it."""
    import streamlit.components.v1 as components
    components.html(_SPLASH_HTML, height=0)


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
