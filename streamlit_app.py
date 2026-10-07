"""HALYARD — cable-fatigue cost of grid-frequency support on a floating offshore wind turbine.

Streamlit Community Cloud entry point. Every number shown is computed live by the physics
engine in physics/ (reduced-order, tagged) or read from ingested high-fidelity data. When
the effect is not real under the chosen conditions, the Phase-1 gate says so plainly.
"""
from __future__ import annotations

import streamlit as st

from app import tabs, theme
from app.runner import sidebar_inputs
from app.theme import fidelity_tag

# "auto" keeps the sidebar expanded on desktop but auto-collapses it on phones, so the
# controls don't cover a small screen on first load (the biggest mobile win). The explicit
# compact-layout toggle in the sidebar handles the rest of the responsive behaviour.
st.set_page_config(page_title="HALYARD", layout="wide", initial_sidebar_state="auto")
theme.inject_css()


def main():
    # Simple, clean loading page — shown once per session (covers the first load, never
    # flickers on ordinary slider reruns). It HOLDS until the page is actually ready: the
    # overlay is emitted here at full opacity, and splash_hide() is emitted at the very END
    # of this run (below). Streamlit streams elements as they are produced, so the overlay
    # stays up for the whole page build and then fades — no fixed timer that could reveal a
    # half-rendered page.
    first_load = not st.session_state.get("_booted")
    if first_load:
        st.html(theme.boot_splash())      # st.html = direct DOM (not an iframe), so it covers the viewport
        st.session_state["_booted"] = True

    p = sidebar_inputs()
    # Read the compact-layout toggle from session_state (set by its sidebar widget's key),
    # NOT from p — p is the cache key for every @st.cache_data sim, and a UI preference must
    # not bust those caches or trigger a recompute when the layout is toggled.
    mobile = bool(st.session_state.get("_mobile_view", False))
    if mobile:
        theme.inject_mobile_css()      # force the compact layout at any viewport width

    st.markdown("# HALYARD")
    st.markdown(
        fidelity_tag("reduced") +
        " &nbsp; Same grid service, measurably different hang-off fatigue, at a quantified "
        "energy-capture trade — <em>the model chain indicates</em>.",
        unsafe_allow_html=True)

    names = ["Overview", "Grid & control", "Platform & cable", "Phase-1 gate",
             "Recovery sweep", "Comparison", "Fatigue detail",
             "High-fidelity ingest", "Validation"]
    renderers = [tabs.tab_overview, tabs.tab_grid_control, tabs.tab_platform_cable,
                 tabs.tab_phase1, tabs.tab_sweep, tabs.tab_comparison, tabs.tab_fatigue,
                 tabs.tab_ingest, tabs.tab_validation]
    # Section navigation. Only the SELECTED section runs on each rerun. st.tabs would execute
    # all nine section bodies on every rerun — including several coupled 6-DOF simulations
    # (Grid, Platform, Phase-1, Comparison, Fatigue) and the BEM hydro build (Validation) — so
    # the first cold load overwhelmed resource-limited hosting. This keeps first paint light.
    # In compact/mobile layout the nine-chip radio becomes a dropdown (far tidier on a phone).
    # The last choice is remembered in _nav so switching widget type keeps you on the section.
    prev = st.session_state.get("_nav", names[0])
    idx = names.index(prev) if prev in names else 0
    if mobile:
        section = st.selectbox("Section", names, index=idx, label_visibility="collapsed",
                               key="_nav_select")
    else:
        section = st.radio("Section", names, index=idx, horizontal=True,
                           label_visibility="collapsed", key="_nav_radio")
    st.session_state["_nav"] = section
    st.markdown("")
    try:
        with st.spinner(f"Computing {section}…"):
            renderers[names.index(section)](p)
    finally:
        # Dismiss the loading page now that everything above has rendered (hold-until-ready).
        # In a finally so a section error can never leave the overlay stuck over the page.
        if first_load:
            st.html(theme.splash_hide())


if __name__ == "__main__":
    main()
