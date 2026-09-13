"""Every technical drawing / diagram / 3-D figure builds from the real objects (§8).

These are the CAD/blueprint/schematic figures wired into the app tabs. The test guarantees
each returns a Plotly go.Figure, carries content (traces and/or blueprint furniture), holds no
non-finite coordinates, and serialises to JSON (what Streamlit does to render). It is the
regression net for the visual layer — figures are built ONLY from computed geometry.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from analysis.case import CaseResult
from models.dynamic_cable import REFERENCE_CABLE
from models.iea15mw import IEA15MW
from models.volturnus_s import VOLTURNUS_S
from physics.cable import (REGIONS, build_quasistatic_family, region_stress_timeseries,
                           solve_static_shape)
from physics.fatigue import FatigueParams, rainflow_damage

from app import (dwg_arrangement, dwg_cable, dwg_diagrams, dwg_dlc, dwg_fatigue,
                 dwg_mooring, dwg_nacelle, viz3d_pro)


@pytest.fixture(scope="module")
def shape():
    return solve_static_shape(REFERENCE_CABLE)


@pytest.fixture(scope="module")
def fatigue_case():
    """A real per-region fatigue set + S-N curve, built without a full simulation."""
    fam = build_quasistatic_family()
    t = np.linspace(0.0, 300.0, 3000)
    surge = 20.0 + 0.5 * np.sin(2 * np.pi * t / 120.0) + 0.2 * np.sin(2 * np.pi * t / 12.0)
    pitch = np.radians(5.5 + 0.3 * np.sin(2 * np.pi * t / 28.0))
    fp = FatigueParams()
    stress, curv, tens, fat = {}, {}, {}, {}
    for reg in REGIONS:
        s, k, T = region_stress_timeseries(fam, reg, surge, pitch, daf=1.2, hangoff_stick=True)
        stress[reg], curv[reg], tens[reg] = s, k, T
        fat[reg] = rainflow_damage(s, float(t[-1] - t[0]), fp)
    case = CaseResult(sim=None, stress_Pa=stress, curvature=curv, tension_N=tens,
                      fatigue=fat, t_count=t, daf=1.2, hangoff_stick=True)
    return case, fp.sn


def _assert_ok(fig, name, min_content=1):
    assert isinstance(fig, go.Figure), f"{name}: not a go.Figure"
    n_content = len(fig.data) + len(fig.layout.shapes or []) + len(fig.layout.annotations or [])
    assert n_content >= min_content, f"{name}: empty figure ({n_content} elements)"
    for tr in fig.data:
        for attr in ("x", "y", "z"):
            v = getattr(tr, attr, None)
            if v is None:
                continue
            arr = np.array([c for c in v if isinstance(c, (int, float))], dtype=float)
            if arr.size:
                assert np.all(np.isfinite(arr)), f"{name}: non-finite {attr} coords"
    fig.to_json()   # Streamlit serialises to JSON to render


def test_arrangement(shape):
    _assert_ok(dwg_arrangement.ga_elevation(shape, REFERENCE_CABLE, VOLTURNUS_S, IEA15MW),
               "ga_elevation", min_content=30)
    _assert_ok(dwg_arrangement.plan_view(REFERENCE_CABLE, VOLTURNUS_S, IEA15MW),
               "plan_view", min_content=30)


def test_cable(shape):
    _assert_ok(dwg_cable.cross_section_datasheet(REFERENCE_CABLE),
               "cross_section_datasheet", min_content=30)
    _assert_ok(dwg_cable.lazywave_config(shape, REFERENCE_CABLE),
               "lazywave_config", min_content=30)
    _assert_ok(dwg_cable.bend_stiffener_detail(REFERENCE_CABLE),
               "bend_stiffener_detail", min_content=20)


def test_mooring():
    _assert_ok(dwg_mooring.mooring_profile(VOLTURNUS_S), "mooring_profile", min_content=30)
    _assert_ok(dwg_mooring.restoring_curve(VOLTURNUS_S), "restoring_curve")
    _assert_ok(dwg_mooring.mooring_plan(VOLTURNUS_S), "mooring_plan", min_content=30)


def test_nacelle():
    _assert_ok(dwg_nacelle.nacelle_cutaway(IEA15MW), "nacelle_cutaway", min_content=30)
    _assert_ok(dwg_nacelle.nacelle_cutaway(), "nacelle_cutaway(default)", min_content=30)


def test_diagrams():
    for name in ("coupling_block", "control_loop", "hydro_block", "pipeline_flow"):
        _assert_ok(getattr(dwg_diagrams, name)(), name, min_content=20)


def test_dlc():
    _assert_ok(dwg_dlc.dlc_matrix_diagram(), "dlc_matrix_diagram", min_content=20)
    _assert_ok(dwg_dlc.metocean_scatter_diagram(), "metocean_scatter_diagram")


def test_fatigue(fatigue_case):
    case, sn = fatigue_case
    hf = case.fatigue["hang_off"]
    _assert_ok(dwg_fatigue.sn_diagram(hf, sn), "sn_diagram")
    _assert_ok(dwg_fatigue.damage_accumulation(case), "damage_accumulation")
    _assert_ok(dwg_fatigue.rainflow_matrix(hf), "rainflow_matrix")


def test_viz3d_pro(shape):
    fig = viz3d_pro.system_figure_pro(shape, REFERENCE_CABLE, VOLTURNUS_S,
                                      color_label="curvature (1/m)")
    _assert_ok(fig, "system_figure_pro", min_content=10)
    # stress-coloured path (as the Platform tab calls it)
    field = np.linspace(-30.0, -2.0, len(shape.x))
    fig2 = viz3d_pro.system_figure_pro(shape, REFERENCE_CABLE, VOLTURNUS_S,
                                       stress_along=field, color_label="log damage",
                                       turbine=IEA15MW)
    _assert_ok(fig2, "system_figure_pro(stress)", min_content=10)
    assert isinstance(viz3d_pro.nacelle_assembly((2.0, 0.0, 150.0)), list)
