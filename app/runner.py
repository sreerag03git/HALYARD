"""Config assembly, sidebar inputs, and cached runs for the HALYARD app.

Kept separate from the tab renderers to avoid circular imports. All runs are deterministic
(seeded) and cached by their primitive parameters, so repeat views are instant.
"""
from __future__ import annotations

import streamlit as st

from analysis.comparison import run_comparison
from analysis.config_io import list_cases, load_case
from analysis.phase1 import Phase1Thresholds, run_phase1
from analysis.sweep import run_sweep
from analysis.case import run_case
from physics.cable import build_quasistatic_family
from physics.controller import EventSchedule, RecoveryParams, SupportParams
from physics.fatigue import FatigueParams, SNCurve
from physics.grid import GridModel
from physics.simulation import SimConfig


@st.cache_resource(show_spinner="Solving the lazy-wave cable family…")
def get_family():
    return build_quasistatic_family()


def make_config(p: dict, enable_support: bool = True,
                recovery: RecoveryParams | None = None,
                t_end: float | None = None) -> SimConfig:
    grid = GridModel(H_sys_s=p["H_sys"], D_load=p["D_load"], S_base_MW=p["S_base"])
    support = SupportParams(dP_max=p["dP_max"], H_wt_s=p["H_wt"], R_droop=p["R_droop"])
    schedule = EventSchedule(support_window_s=p["support_window"])
    rp = recovery or RecoveryParams(kind="shaped", tau_rec_s=p["tau_rec"],
                                    rate_rec_pu_s=p["rate_rec"])
    return SimConfig(wind_ms=p["wind"], Hs_m=p["Hs"], Tp_s=p["Tp"], wave_seed=p["seed"],
                     t_end_s=t_end or p["t_end"], dt_s=0.02, p_load_pu=p["p_load"],
                     wind_capacity_MW=p["wind_cap"], enable_support=enable_support,
                     grid=grid, support=support, recovery=rp, schedule=schedule)


def make_fatigue(p: dict) -> FatigueParams:
    return FatigueParams(sn=SNCurve(m1=p["m1"], m2=max(p["m1"] + 2.0, 5.0),
                                    sigma_u_MPa=p["sigma_u"]),
                         mean_stress=p["mean_stress"], DFF=p["DFF"])


def make_thresholds(p: dict) -> Phase1Thresholds:
    return Phase1Thresholds(added_damage_frac_min=p["thr_added"],
                            recovery_share_min=p["thr_recov"],
                            min_slow_stress_amp_MPa=p["thr_amp"])


@st.cache_data(show_spinner=False)
def cached_case(p: dict, enable_support: bool):
    return run_case(make_config(p, enable_support=enable_support), get_family(),
                    make_fatigue(p), p["daf"], p["hangoff_stick"])


@st.cache_data(show_spinner=False)
def cached_phase1(p: dict):
    return run_phase1(make_config(p, enable_support=True), thresholds=make_thresholds(p),
                      family=get_family(), fatigue_params=make_fatigue(p), daf=p["daf"],
                      hangoff_stick=p["hangoff_stick"])


@st.cache_data(show_spinner=False)
def cached_comparison(p: dict):
    rp = RecoveryParams(kind="shaped", tau_rec_s=p["tau_rec"], rate_rec_pu_s=p["rate_rec"])
    return run_comparison(make_config(p, enable_support=True), halyard_recovery=rp,
                          family=get_family(), fatigue_params=make_fatigue(p), daf=p["daf"],
                          hangoff_stick=p["hangoff_stick"])


@st.cache_data(show_spinner=False)
def cached_annualize(p: dict):
    from analysis.annualize import annualized_life
    return annualized_life(make_config(p, enable_support=True, t_end=360.0),
                           events_per_year=p["events_per_year"], family=get_family(),
                           fatigue_params=make_fatigue(p), daf=p["daf"],
                           hangoff_stick=p["hangoff_stick"])


@st.cache_data(show_spinner=False)
def cached_sweep(p: dict):
    # Stage A is an explicit offline optimiser: a compact 4x3 recovery grid at a shorter
    # duration keeps it responsive while still resolving the trade-off and Pareto front.
    return run_sweep(make_config(p, enable_support=True, t_end=360.0),
                     tau_values=[3.0, 10.0, 20.0, 30.0],
                     rate_values=[0.02, 0.06, 0.15],
                     family=get_family(), fatigue_params=make_fatigue(p), daf=p["daf"],
                     hangoff_stick=p["hangoff_stick"])


def sidebar_inputs() -> dict:
    st.sidebar.markdown("### HALYARD")
    st.sidebar.caption("Cable-fatigue cost of grid-frequency support · IEA-15MW / VolturnUS-S")

    cases = list_cases()
    default_idx = cases.index("default") if "default" in cases else 0
    case_name = st.sidebar.selectbox("Committed case", cases, index=default_idx)
    base = load_case(case_name)
    b = base.raw

    st.sidebar.markdown("**Grid** — reduced-order SFR")
    H_sys = st.sidebar.slider("System inertia H_sys (s)", 1.0, 8.0,
                              float(b["grid"]["H_sys_s"]), 0.5,
                              help="Low-inertia (small H) → steeper RoCoF, deeper nadir; the effect grows as grids weaken.")
    p_load = st.sidebar.slider("Loss of infeed ΔP_load (pu)", 0.02, 0.20,
                               float(b["sim"]["p_load_pu"]), 0.01)

    st.sidebar.markdown("**Sea state** — JONSWAP, seeded")
    Hs = st.sidebar.slider("Significant wave height Hs (m)", 0.5, 5.0,
                           float(b["sim"]["Hs_m"]), 0.25)
    Tp = st.sidebar.slider("Peak period Tp (s)", 5.0, 16.0, float(b["sim"]["Tp_s"]), 0.5)
    wind = st.sidebar.slider("Wind speed (m/s, ≤ rated 10.59)", 6.0, 10.5,
                             float(b["sim"]["wind_ms"]), 0.5)

    st.sidebar.markdown("**Support** — grid-mandated, not shaped")
    dP_max = st.sidebar.slider("Support magnitude limit ΔP_max (pu)", 0.05, 0.25,
                               float(b["support"]["dP_max"]), 0.01)
    H_wt = st.sidebar.slider("Emulated inertia H_wt (s)", 2.0, 15.0,
                             float(b["support"]["H_wt_s"]), 1.0)
    support_window = st.sidebar.slider("Support window (s)", 5.0, 30.0,
                                       float(b["schedule"]["support_window_s"]), 1.0)

    st.sidebar.markdown("**Recovery** — the design lever (§5.3)")
    tau_rec = st.sidebar.slider("Recovery time constant τ_rec (s)", 2.0, 30.0,
                                float(b["recovery"]["tau_rec_s"]), 1.0,
                                help="Larger = gentler recovery.")
    rate_rec = st.sidebar.slider("Recovery slew limit (pu/s)", 0.01, 0.20,
                                 float(b["recovery"]["rate_rec_pu_s"]), 0.01)

    with st.sidebar.expander("Fatigue & sensitivity"):
        m1 = st.slider("S-N slope m", 3.0, 5.0, float(b["fatigue"]["sn_curve"]["m1"]), 0.5,
                       help="Sensitivity toggle — the qualitative conclusion should survive.")
        mean_stress = st.selectbox("Mean-stress correction", ["goodman", "gerber", "none"])
        DFF = st.slider("Design Fatigue Factor", 1.0, 10.0, float(b["fatigue"]["DFF"]), 1.0)
        daf = st.slider("Dynamic amplification factor", 1.0, 1.6, float(b["cable"]["daf"]), 0.05)
        hangoff_stick = st.checkbox("Hang-off stick regime (no-slip lever)", value=True,
                                    help="Off = slip (wire-radius) lower bound at the hang-off.")

    with st.sidebar.expander("Committed Phase-1 thresholds"):
        st.caption("Pre-registered in the case YAML (committed before results).")
        thr_added = st.number_input("Added-damage fraction X", 0.0, 0.5,
                                    float(b["phase1_thresholds"]["added_damage_frac_min"]), 0.01)
        thr_recov = st.number_input("Recovery share Y", 0.0, 1.0,
                                    float(b["phase1_thresholds"]["recovery_share_min"]), 0.05)
        thr_amp = st.number_input("Min slow-stress amplitude (MPa)", 0.0, 10.0,
                                  float(b["phase1_thresholds"]["min_slow_stress_amp_MPa"]), 0.1)

    with st.sidebar.expander("Advanced / reproducibility"):
        seed = st.number_input("Wave seed", 0, 99999, int(b["sim"]["wave_seed"]))
        t_end = st.slider("Simulation length (s)", 300, 700, int(b["sim"]["t_end_s"]), 50)
        S_base = st.number_input("System base (MW)", 500.0, 30000.0,
                                 float(b["grid"]["S_base_MW"]), 500.0)
        wind_cap = st.number_input("Supporting wind capacity (MW)", 100.0, 5000.0,
                                   float(b["sim"]["wind_capacity_MW"]), 100.0)

    return dict(case_name=case_name, H_sys=H_sys, D_load=float(b["grid"]["D_load"]),
                S_base=float(S_base), p_load=p_load, Hs=Hs, Tp=Tp, wind=wind, dP_max=dP_max,
                H_wt=H_wt, R_droop=float(b["support"]["R_droop"]),
                support_window=support_window, tau_rec=tau_rec, rate_rec=rate_rec, m1=m1,
                sigma_u=1400.0, mean_stress=mean_stress, DFF=DFF, daf=daf,
                hangoff_stick=hangoff_stick, thr_added=thr_added, thr_recov=thr_recov,
                thr_amp=thr_amp, seed=int(seed), t_end=float(t_end), wind_cap=float(wind_cap),
                events_per_year=base.events_per_year, occurrence=base.sea_state_occurrence,
                case_description=base.description)
