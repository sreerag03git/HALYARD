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
    # Neat, on-brand loading page — rendered once per session so it covers the cold-start
    # boot (and the first heavy compute) but never flickers on ordinary slider reruns.
    if not st.session_state.get("_booted"):
        st.html(theme.boot_splash())      # st.html (direct DOM, not an iframe) so the overlay covers the viewport
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
    with st.spinner(f"Computing {section}…"):
        renderers[names.index(section)](p)


if __name__ == "__main__":
    main()
