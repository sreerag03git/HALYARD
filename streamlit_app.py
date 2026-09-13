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
    t = st.tabs(names)
    with t[0]:
        tabs.tab_overview(p)
    with t[1]:
        tabs.tab_grid_control(p)
    with t[2]:
        tabs.tab_platform_cable(p)
    with t[3]:
        tabs.tab_phase1(p)
    with t[4]:
        tabs.tab_sweep(p)
    with t[5]:
        tabs.tab_comparison(p)
    with t[6]:
        tabs.tab_fatigue(p)
    with t[7]:
        tabs.tab_ingest(p)
    with t[8]:
        tabs.tab_validation(p)


if __name__ == "__main__":
    main()
