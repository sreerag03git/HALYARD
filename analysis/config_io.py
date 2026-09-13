"""Config-driven cases (§11): one YAML fully defines a case.

A case YAML names the (frozen) turbine/platform references and sets grid, sea-state,
control, recovery, seeds, fatigue, and the PRE-REGISTERED Phase-1 thresholds. No physical
magic numbers live in code — they are here or in the reference-model modules. This loader
builds the runtime objects and keeps the raw dict for the run log (reproducibility).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

from analysis.phase1 import Phase1Thresholds
from physics.controller import EventSchedule, RecoveryParams, SupportParams
from physics.fatigue import FatigueParams, SNCurve
from physics.grid import GridModel
from physics.simulation import SimConfig

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


@dataclass
class Case:
    name: str
    description: str
    sim: SimConfig
    thresholds: Phase1Thresholds
    fatigue: FatigueParams
    daf: float
    hangoff_stick: bool
    events_per_year: float
    sea_state_occurrence: float
    raw: dict = field(default_factory=dict)


def load_case(path_or_name: str) -> Case:
    """Load a case from a YAML path or a bare name in configs/."""
    path = path_or_name
    if not os.path.exists(path):
        cand = os.path.join(CONFIG_DIR, path_or_name)
        if not cand.endswith((".yaml", ".yml")):
            cand += ".yaml"
        path = cand
    with open(path, "r") as fh:
        d = yaml.safe_load(fh)
    return case_from_dict(d)


def case_from_dict(d: dict) -> Case:
    g = d.get("grid", {})
    grid = GridModel(
        H_sys_s=g.get("H_sys_s", 2.5), D_load=g.get("D_load", 1.0),
        R_sys=g.get("R_sys", 0.05), T_gov_s=g.get("T_gov_s", 8.0),
        reserve_pu=g.get("reserve_pu", 0.12), T_rocof_s=g.get("T_rocof_s", 0.5),
        S_base_MW=g.get("S_base_MW", 3000.0))
    sp = d.get("support", {})
    support = SupportParams(
        H_wt_s=sp.get("H_wt_s", 6.0), R_droop=sp.get("R_droop", 0.05),
        deadband_hz=sp.get("deadband_hz", 0.015), dP_max=sp.get("dP_max", 0.10),
        rate_pu_s=sp.get("rate_pu_s", 0.50), tau_s=sp.get("tau_s", 0.20))
    rc = d.get("recovery", {})
    recovery = RecoveryParams(
        kind=rc.get("kind", "shaped"), tau_rec_s=rc.get("tau_rec_s", 12.0),
        rate_rec_pu_s=rc.get("rate_rec_pu_s", 0.02), Kp_rec=rc.get("Kp_rec", 0.6),
        dP_dip_max=rc.get("dP_dip_max", 0.12), shape_exponent=rc.get("shape_exponent", 1.0))
    sc = d.get("schedule", {})
    schedule = EventSchedule(
        t_event_s=sc.get("t_event_s", 100.0), support_window_s=sc.get("support_window_s", 10.0))
    s = d.get("sim", {})
    sim = SimConfig(
        wind_ms=s.get("wind_ms", 9.0), Hs_m=s.get("Hs_m", 2.0), Tp_s=s.get("Tp_s", 8.0),
        wave_seed=s.get("wave_seed", 1234), t_end_s=s.get("t_end_s", 350.0),
        dt_s=s.get("dt_s", 0.025), t_discard_s=s.get("t_discard_s", 100.0),
        p_load_pu=s.get("p_load_pu", 0.10), wind_capacity_MW=s.get("wind_capacity_MW", 900.0),
        enable_support=s.get("enable_support", True), beta_deg=s.get("beta_deg", 0.0),
        grid=grid, support=support, recovery=recovery, schedule=schedule)
    th = d.get("phase1_thresholds", {})
    thresholds = Phase1Thresholds(
        added_damage_frac_min=th.get("added_damage_frac_min", 0.05),
        recovery_share_min=th.get("recovery_share_min", 0.20),
        min_slow_stress_amp_MPa=th.get("min_slow_stress_amp_MPa", 0.5),
        slow_cutoff_hz=th.get("slow_cutoff_hz", 0.03))
    ft = d.get("fatigue", {})
    sn = ft.get("sn_curve", {})
    # float() coercion guards against PyYAML parsing e.g. "1.0e6" as a string.
    fatigue = FatigueParams(
        sn=SNCurve(m1=float(sn.get("m1", 3.0)), log_a1=float(sn.get("log_a1", 11.764)),
                   m2=float(sn.get("m2", 5.0)), log_a2=float(sn.get("log_a2", 15.606)),
                   N_knee=float(sn.get("N_knee", 1e6)),
                   sigma_u_MPa=float(sn.get("sigma_u_MPa", 1400.0))),
        mean_stress=ft.get("mean_stress", "goodman"), DFF=float(ft.get("DFF", 3.0)))
    return Case(
        name=d.get("name", "case"), description=d.get("description", ""),
        sim=sim, thresholds=thresholds, fatigue=fatigue,
        daf=d.get("cable", {}).get("daf", 1.2),
        hangoff_stick=d.get("cable", {}).get("hangoff_stick", True),
        events_per_year=d.get("annualization", {}).get("events_per_year", 500.0),
        sea_state_occurrence=d.get("annualization", {}).get("sea_state_occurrence", 1.0),
        raw=d)


def list_cases() -> list:
    if not os.path.isdir(CONFIG_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml"))
