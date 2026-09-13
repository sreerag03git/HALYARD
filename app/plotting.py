"""Result plots — engineering figures with consistent, colour-blind-safe styling (§8).

Every figure is computed from model output or real data; axes are labelled with units and
legends are always present. The three controllers keep the same colours across all charts.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from app import theme
from app.theme import ACCENT, C_HALYARD, C_INCUMBENT, C_NO_SUPPORT, GREY, WARN, GOOD, INK
from models.iea15mw import IEA15MW


def _mode_shading(fig, sim):
    """Shade the support (light amber) and recovery (light blue) windows."""
    t = sim.t
    for mval, color in [(1, "rgba(230,159,0,0.08)"), (2, "rgba(58,90,120,0.08)")]:
        idx = np.where(sim.mode == mval)[0]
        if len(idx):
            fig.add_vrect(x0=t[idx[0]], x1=t[idx[-1]], fillcolor=color, line_width=0,
                          layer="below")


def frequency_plot(sim):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.t, y=sim.freq_hz, name="System frequency",
                             line=dict(color=ACCENT, width=2)))
    _mode_shading(fig, sim)
    fig.add_hline(y=sim.config.grid.f0_Hz, line=dict(color=GREY, dash="dot", width=1))
    return theme.plotly_layout(fig, "Grid frequency (support = amber, recovery = blue)",
                               xtitle="time (s)", ytitle="frequency (Hz)")


def rocof_plot(sim):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.t, y=sim.rocof_hz_s, name="RoCoF (measured)",
                             line=dict(color=WARN, width=1.6)))
    _mode_shading(fig, sim)
    return theme.plotly_layout(fig, "Rate of change of frequency",
                               xtitle="time (s)", ytitle="RoCoF (Hz/s)")


def rotor_plot(sim):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.t, y=sim.omega_rads, name="rotor speed omega",
                             line=dict(color=ACCENT, width=2)))
    fig.add_hline(y=IEA15MW.omega_min_rads, line=dict(color=WARN, dash="dash", width=1),
                  annotation_text="omega_min", annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=sim.t, y=sim.ke_MJ, name="rotor KE (MJ)", yaxis="y2",
                             line=dict(color=GREY, width=1.4, dash="dot")))
    _mode_shading(fig, sim)
    fig.update_layout(yaxis2=dict(title="KE (MJ)", overlaying="y", side="right",
                                  showgrid=False))
    return theme.plotly_layout(fig, "Rotor speed & kinetic energy (the support reservoir)",
                               xtitle="time (s)", ytitle="omega (rad/s)")


def power_thrust_plot(sim):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.t, y=sim.P_elec_W / 1e6, name="electrical power (MW)",
                             line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=sim.t, y=sim.thrust_N / 1e6, name="rotor thrust (MN)",
                             yaxis="y2", line=dict(color=WARN, width=1.8)))
    _mode_shading(fig, sim)
    fig.update_layout(yaxis2=dict(title="thrust (MN)", overlaying="y", side="right",
                                  showgrid=False))
    return theme.plotly_layout(fig, "Electrical power & rotor thrust (the coupling)",
                               xtitle="time (s)", ytitle="power (MW)")


def platform_plot(sim):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.t, y=sim.surge_m, name="surge (m)",
                             line=dict(color=ACCENT, width=1.6)))
    fig.add_trace(go.Scatter(x=sim.t, y=sim.pitch_deg, name="pitch (deg)", yaxis="y2",
                             line=dict(color=WARN, width=1.6)))
    _mode_shading(fig, sim)
    fig.update_layout(yaxis2=dict(title="pitch (deg)", overlaying="y", side="right",
                                  showgrid=False))
    return theme.plotly_layout(fig, "Floating-platform motion (surge & pitch)",
                               xtitle="time (s)", ytitle="surge (m)")


def platform_6dof_plot(sim):
    """All available platform DOFs (6-DOF engine): translations and rotations, two axes."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.t, y=sim.surge_m, name="surge (m)",
                             line=dict(color=ACCENT, width=1.6)))
    if sim.heave_m is not None:
        fig.add_trace(go.Scatter(x=sim.t, y=sim.heave_m, name="heave (m)",
                                 line=dict(color="#2E6B4F", width=1.4)))
    if sim.sway_m is not None and np.abs(sim.sway_m).max() > 1e-4:
        fig.add_trace(go.Scatter(x=sim.t, y=sim.sway_m, name="sway (m)",
                                 line=dict(color="#7FA8C9", width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=sim.t, y=sim.pitch_deg, name="pitch (deg)", yaxis="y2",
                             line=dict(color=WARN, width=1.6)))
    if sim.roll_deg is not None and np.abs(sim.roll_deg).max() > 1e-4:
        fig.add_trace(go.Scatter(x=sim.t, y=sim.roll_deg, name="roll (deg)", yaxis="y2",
                                 line=dict(color="#C98F3A", width=1.2, dash="dot")))
    _mode_shading(fig, sim)
    fig.update_layout(yaxis2=dict(title="rotation (deg)", overlaying="y", side="right",
                                  showgrid=False))
    return theme.plotly_layout(fig, "Floating-platform motion — 6-DOF (BEM Cummins hydro)",
                               xtitle="time (s)", ytitle="translation (m)")


def stress_plot(t, stress_by_region, focus="hang_off"):
    fig = go.Figure()
    for reg, sig in stress_by_region.items():
        width = 2.2 if reg == focus else 1.0
        color = theme.REGION_COLORS.get(reg, GREY)
        fig.add_trace(go.Scatter(x=t, y=np.asarray(sig) / 1e6, name=reg.replace("_", "-"),
                                 line=dict(color=color, width=width),
                                 opacity=1.0 if reg == focus else 0.55))
    return theme.plotly_layout(fig, "Cable stress by region (combined signal)",
                               xtitle="time (s)", ytitle="stress (MPa)")


def psd_overlap_plot(p1):
    """Phase-1 evidence: PSD of hang-off stress, wave-only vs +support, + modal bands."""
    fig = go.Figure()
    f = p1.psd_freq
    fig.add_trace(go.Scatter(x=f, y=p1.psd_wave_only, name="wave only",
                             line=dict(color=C_NO_SUPPORT, width=1.8)))
    fig.add_trace(go.Scatter(x=f, y=p1.psd_with_support, name="wave + frequency support",
                             line=dict(color=ACCENT, width=1.8)))
    positions = {"surge": "top left", "pitch": "top right"}
    for name, fb in p1.modal_bands_hz.items():
        fig.add_vline(x=fb, line=dict(color=WARN, dash="dot", width=1),
                      annotation_text=f"{name} mode",
                      annotation_position=positions.get(name, "top"),
                      annotation_font=dict(size=10, color=WARN))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return theme.plotly_layout(fig, "Hang-off stress PSD — spectral overlap (Phase-1)",
                               xtitle="frequency (Hz)", ytitle="PSD (MPa^2/Hz)")


def rainflow_hist_plot(fat):
    from physics.fatigue import rainflow_histogram
    centres, counts = rainflow_histogram(fat)
    fig = go.Figure()
    if len(centres):
        fig.add_trace(go.Bar(x=centres, y=counts, marker_color=ACCENT, name="cycles"))
    return theme.plotly_layout(fig, "Rainflow range histogram (hang-off)",
                               xtitle="stress range (MPa)", ytitle="cycle count",
                               legend=False)


def sn_plot(fat, sn):
    """S-N curve with the counted (mean-corrected) cycles plotted on it."""
    S = np.logspace(0, 3.2, 200)
    N = sn.cycles_to_failure(S)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=N, y=S, name="S-N curve (DNV-style)",
                             line=dict(color=INK, width=2)))
    if len(fat.S_eq_MPa):
        fig.add_trace(go.Scatter(x=fat.N_i, y=fat.S_eq_MPa, mode="markers",
                                 name="counted cycles (S_eq)",
                                 marker=dict(color=ACCENT, size=6, opacity=0.6)))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return theme.plotly_layout(fig, "S-N curve with counted cycles (hang-off)",
                               xtitle="cycles to failure N", ytitle="equivalent range S_eq (MPa)")


def pareto_plot(sweep):
    fig = go.Figure()
    feas = sweep.feasible()
    infeas = [p for p in sweep.points if not (p.kpis_ok and p.stable)]
    if infeas:
        fig.add_trace(go.Scatter(
            x=[100 * p.aep_penalty_frac for p in infeas],
            y=[p.hangoff_damage for p in infeas], mode="markers", name="infeasible (grid/stability)",
            marker=dict(color=GREY, size=7, symbol="x", opacity=0.6)))
    if feas:
        fig.add_trace(go.Scatter(
            x=[100 * p.aep_penalty_frac for p in feas],
            y=[p.hangoff_damage for p in feas], mode="markers", name="feasible",
            marker=dict(color=ACCENT, size=9,
                        line=dict(color="white", width=1)),
            text=[f"tau={p.tau_rec_s:.0f}s rate={p.rate_rec_pu_s:.2f}" for p in feas]))
    par = sweep.pareto()
    if par:
        fig.add_trace(go.Scatter(
            x=[100 * p.aep_penalty_frac for p in par],
            y=[p.hangoff_damage for p in par], mode="lines+markers", name="Pareto front",
            line=dict(color=WARN, width=2)))
    fig.add_hline(y=sweep.incumbent_damage, line=dict(color=C_INCUMBENT, dash="dash"),
                  annotation_text="incumbent", annotation_position="top left")
    return theme.plotly_layout(fig, "Recovery-shape trade-off: hang-off fatigue vs AEP penalty",
                               xtitle="AEP penalty (%)", ytitle="hang-off damage per window")


def comparison_bars(cmp):
    labels = ["No support", "Incumbent", "HALYARD"]
    dmg = [cmp.no_support.case.hangoff_damage, cmp.incumbent.case.hangoff_damage,
           cmp.halyard.case.hangoff_damage]
    colors = [C_NO_SUPPORT, C_INCUMBENT, C_HALYARD]
    fig = go.Figure(go.Bar(x=labels, y=dmg, marker_color=colors,
                           text=[f"{d:.2e}" for d in dmg], textposition="outside"))
    return theme.plotly_layout(fig, "Hang-off fatigue damage per window (three controllers)",
                               xtitle="", ytitle="Miner damage per window", legend=False)


def region_damage_bars(case):
    from physics.cable import REGIONS
    labels = [r.replace("_", "-") for r in REGIONS]
    dmg = [case.fatigue[r].damage for r in REGIONS]
    colors = [theme.REGION_COLORS[r] for r in REGIONS]
    fig = go.Figure(go.Bar(x=labels, y=dmg, marker_color=colors,
                           text=[f"{d:.2e}" for d in dmg], textposition="outside"))
    fig.update_yaxes(type="log")
    return theme.plotly_layout(fig, "Fatigue damage by cable region (hang-off concentration)",
                               xtitle="", ytitle="Miner damage per window (log)", legend=False)
