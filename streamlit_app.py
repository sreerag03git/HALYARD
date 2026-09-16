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

st.set_page_config(page_title="HALYARD", layout="wide", initial_sidebar_state="expanded")
theme.inject_css()


def main():
    p = sidebar_inputs()
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
    section = st.radio("Section", names, horizontal=True, label_visibility="collapsed",
                       key="_section")
    st.markdown("")
    renderers[names.index(section)](p)


if __name__ == "__main__":
    main()
