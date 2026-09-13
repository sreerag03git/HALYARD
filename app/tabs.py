"""Tab renderers for the HALYARD app. Each tab mirrors a stage of the pipeline (§8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import drawings, plotting, viz3d
from app.runner import (cached_case, cached_comparison, cached_phase1, cached_sweep,
                        get_family, make_fatigue)
from app.theme import fidelity_tag, what_this_shows, GOOD, WARN
from analysis import validation as val
from physics.cable import REGIONS, _region_indices, solve_static_shape
from models.dynamic_cable import REFERENCE_CABLE
from models.iea15mw import IEA15MW
from models.volturnus_s import VOLTURNUS_S


def _damage_field(shape, cable, region_damage: dict) -> np.ndarray:
    """Map the four analysed regions' damage onto every cable node as a smooth field.

    Each region contributes a Gaussian bump (in arc length) scaled by log-damage, so the
    3-D colour shows where fatigue concentrates (typically the hang-off). Visualization
    only — the quantitative damage lives in the four analysed regions.
    """
    reg = _region_indices(shape, cable)
    s = shape.s
    field = np.full_like(s, -30.0)   # log10 floor
    for name, i in reg.items():
        d = region_damage.get(name, 0.0)
        logd = np.log10(d) if d > 0 else -30.0
        bump = np.exp(-((s - s[i]) / 12.0) ** 2)
        field = np.maximum(field, logd * bump + (-30.0) * (1 - bump))
    return field


def tab_overview(p):
    st.markdown("### The system and the mechanism")
    st.markdown(what_this_shows(
        "Interactive 3-D model built entirely from published geometry, and the causation "
        "chain HALYARD quantifies. What it does not show: any result — see the pipeline tabs."),
        unsafe_allow_html=True)
    fam = get_family()
    shape = fam.base_shape
    st.plotly_chart(drawings.coupling_block_diagram(), use_container_width=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(viz3d.system_figure(shape, REFERENCE_CABLE, VOLTURNUS_S,
                                            color_label="curvature (1/m)"),
                        use_container_width=True)
        st.caption("IEA-15MW rotor + tower on the VolturnUS-S semi, catenary moorings, and the "
                   "solved lazy-wave cable (colour = geometric curvature). Rotate / zoom / pan.")
    with c2:
        st.markdown("**Frozen reference values** (read-only)")
        st.dataframe(pd.DataFrame([
            ("Turbine", "IEA-15-240-RWT (Type-4)"),
            ("Rated power", "15 MW"),
            ("Rotor diameter", "240 m"),
            ("Rated rotor speed", "7.56 rpm"),
            ("Platform", "UMaine VolturnUS-S"),
            ("Water depth", "200 m"),
            ("Surge / pitch period", "120 s / 28 s (fitted)"),
            ("Cable", "66 kV lazy-wave (representative)"),
        ], columns=["Quantity", "Value"]), hide_index=True, use_container_width=True)
        st.caption(f"Aero surface: {IEA15MW.aero.provenance[:78]}…")


def tab_grid_control(p):
    st.markdown("### Grid frequency, support and recovery")
    st.markdown(fidelity_tag("reduced") + what_this_shows(
        "The coupled event: a loss of infeed, the turbine's synthetic-inertia + droop support, "
        "and the shaped recovery — with the genuine secondary dip. What it does not show: "
        "fatigue (later tabs)."), unsafe_allow_html=True)
    case = cached_case(p, True)
    sim = case.sim
    from analysis.kpis import grid_kpis, grid_code_compliant
    k = grid_kpis(sim)
    ok, checks = grid_code_compliant(k)
    cols = st.columns(4)
    cols[0].metric("Frequency nadir", f"{k.nadir_hz:.2f} Hz")
    cols[1].metric("Peak RoCoF", f"{k.max_rocof_hz_s:.3f} Hz/s")
    cols[2].metric("Secondary dip", f"{k.secondary_min_hz:.2f} Hz")
    cols[3].metric("Grid-code", "compliant" if ok else "VIOLATION",
                   delta=None)
    if not ok:
        st.warning("This case does not meet the grid code — the KPI checks: "
                   f"{ {kk: bool(vv) for kk, vv in checks.items()} }")
    st.plotly_chart(plotting.frequency_plot(sim), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(plotting.rotor_plot(sim), use_container_width=True)
    b.plotly_chart(plotting.power_thrust_plot(sim), use_container_width=True)
    st.plotly_chart(plotting.rocof_plot(sim), use_container_width=True)


def tab_platform_cable(p):
    st.markdown("### Platform motion and the dynamic cable")
    st.markdown(fidelity_tag("reduced") + what_this_shows(
        "The thrust transient moves the floating platform (surge + pitch); that motion flexes "
        "the lazy-wave cable, worst at the hang-off. Colour on the 3-D cable = fatigue damage "
        "mapped from the four analysed regions."), unsafe_allow_html=True)
    fam = get_family()
    case = cached_case(p, True)
    sim = case.sim
    a, b = st.columns(2)
    a.plotly_chart(plotting.platform_plot(sim), use_container_width=True)
    b.plotly_chart(plotting.stress_plot(case.t_count, case.stress_Pa), use_container_width=True)
    st.plotly_chart(drawings.lazywave_profile(fam.base_shape, REFERENCE_CABLE),
                    use_container_width=True)
    c, d = st.columns([3, 2])
    dmg = {r: case.fatigue[r].damage for r in REGIONS}
    field = _damage_field(fam.base_shape, REFERENCE_CABLE, dmg)
    with c:
        st.plotly_chart(viz3d.system_figure(fam.base_shape, REFERENCE_CABLE, VOLTURNUS_S,
                                            stress_along=field,
                                            color_label="log₁₀ fatigue damage"),
                        use_container_width=True)
    with d:
        st.plotly_chart(drawings.cross_section(REFERENCE_CABLE), use_container_width=True)


def tab_phase1(p):
    st.markdown("### Phase-1 gate — can the effect even exist here?")
    st.markdown(what_this_shows(
        "Before any saving is claimed, HALYARD checks whether frequency-support control adds "
        "meaningful hang-off fatigue vs the wave-only baseline, using PRE-REGISTERED thresholds. "
        "A NEGLIGIBLE verdict is a legitimate result, not an error."), unsafe_allow_html=True)
    with st.spinner("Running the honesty gate (wave-only vs wave+support)…"):
        r = cached_phase1(p)
    if r.passed:
        st.markdown(f"<h2 class='verdict-pass'>PASS — the effect is present</h2>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<h2 class='verdict-fail'>NEGLIGIBLE under these conditions</h2>",
                    unsafe_allow_html=True)
        st.info("Under these inputs the control-driven cable-fatigue effect is negligible. "
                "The effect grows as grids weaken (lower H_sys), support strengthens, and seas "
                "calm. Stage A/B remain locked — HALYARD does not claim a saving here.")
    cols = st.columns(3)
    th = r.thresholds
    cols[0].metric("Added-damage fraction", f"{r.added_damage_frac*100:.1f}%",
                   f"threshold {th.added_damage_frac_min*100:.0f}%",
                   delta_color="normal" if r.reasons["added_damage"] else "inverse")
    cols[1].metric("Recovery share of slow drift", f"{r.recovery_share*100:.0f}%",
                   f"threshold {th.recovery_share_min*100:.0f}%",
                   delta_color="normal" if r.reasons["recovery_share"] else "inverse")
    cols[2].metric("Slow-drift stress amplitude", f"{r.slow_stress_amp_MPa:.2f} MPa",
                   f"threshold {th.min_slow_stress_amp_MPa:.1f} MPa",
                   delta_color="normal" if r.reasons["slow_amplitude"] else "inverse")
    st.plotly_chart(plotting.psd_overlap_plot(r), use_container_width=True)
    st.caption("Hang-off stress PSD: wave-only vs wave+support, with the platform surge/pitch "
               "modal bands. Overlap near the modal bands + a fatigue-relevant amplitude is what "
               "the gate requires — not spectral coincidence alone.")


def tab_sweep(p):
    st.markdown("### Stage A — recovery-shape sweep (offline optimiser)")
    r = cached_phase1(p)
    if not r.passed:
        st.warning("Locked — the Phase-1 gate returned NEGLIGIBLE for these inputs, so there is "
                   "no fatigue effect to optimise. Strengthen the case (weaker grid / stronger "
                   "support / calmer sea) to unlock.")
        return
    st.markdown(what_this_shows(
        "Sweep the recovery-shape parameters (τ_rec, slew) and minimise hang-off fatigue subject "
        "to grid-code compliance and a stable platform. The relationship is not assumed — the "
        "sweep discovers it (gentler is often better, but a recovery near the 28 s pitch period "
        "can be worse)."), unsafe_allow_html=True)
    if st.button("Run recovery sweep (12 cases)", type="primary"):
        st.session_state["_run_sweep"] = True
    if not st.session_state.get("_run_sweep"):
        st.caption("Press to run the sweep (cached afterwards).")
        return
    with st.spinner("Sweeping recovery shapes…"):
        sw = cached_sweep(p)
    st.plotly_chart(plotting.pareto_plot(sw), use_container_width=True)
    knee = sw.best_tradeoff()
    if knee:
        c = st.columns(3)
        c[0].metric("Best τ_rec", f"{knee.tau_rec_s:.0f} s")
        c[1].metric("Hang-off damage vs incumbent",
                    f"{100*(knee.hangoff_damage-sw.incumbent_damage)/sw.incumbent_damage:+.1f}%")
        c[2].metric("AEP penalty", f"{knee.aep_penalty_frac*100:.2f}%")
    rows = [dict(tau_rec_s=q.tau_rec_s, rate_rec=q.rate_rec_pu_s,
                 hangoff_damage=q.hangoff_damage, aep_pct=100*q.aep_penalty_frac,
                 nadir=q.nadir_hz, secondary=q.secondary_min_hz,
                 compliant=q.kpis_ok, stable=q.stable) for q in sw.points]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def tab_comparison(p):
    st.markdown("### Stage B — three-controller comparison (the headline)")
    r = cached_phase1(p)
    if not r.passed:
        st.warning("Locked — Phase-1 returned NEGLIGIBLE, so there is no fatigue difference to "
                   "compare. HALYARD does not claim a saving here.")
        return
    st.markdown(what_this_shows(
        "No support vs the cited incumbent recovery vs HALYARD's shaped recovery, same wave seed. "
        "The equivalence gate must pass FIRST (both meet the grid code); only then is the fatigue "
        "comparison valid."), unsafe_allow_html=True)
    if st.button("Run three-controller comparison", type="primary"):
        st.session_state["_run_cmp"] = True
    if not st.session_state.get("_run_cmp"):
        st.caption("Press to run (three coupled simulations; cached afterwards).")
        return
    with st.spinner("Running three controllers…"):
        cmp = cached_comparison(p)
    if cmp.equivalent:
        st.markdown(f"<span class='verdict-pass'>Electrically equivalent — both recoveries meet "
                    f"the grid code.</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='verdict-fail'>NOT electrically equivalent — the comparison is "
                    "void.</span> One recovery fails the grid code (see KPIs); shape it to comply "
                    "before comparing fatigue.", unsafe_allow_html=True)
    kdf = pd.DataFrame({
        "KPI": ["nadir (Hz)", "peak RoCoF (Hz/s)", "secondary dip (Hz)", "response energy (MJ)"],
        "Incumbent": [cmp.incumbent.kpis.nadir_hz, cmp.incumbent.kpis.max_rocof_hz_s,
                      cmp.incumbent.kpis.secondary_min_hz, cmp.incumbent.kpis.response_energy_MJ],
        "HALYARD": [cmp.halyard.kpis.nadir_hz, cmp.halyard.kpis.max_rocof_hz_s,
                    cmp.halyard.kpis.secondary_min_hz, cmp.halyard.kpis.response_energy_MJ],
    })
    a, b = st.columns([2, 3])
    a.dataframe(kdf.round(3), hide_index=True, use_container_width=True)
    b.plotly_chart(plotting.comparison_bars(cmp), use_container_width=True)
    if cmp.equivalent:
        c = st.columns(2)
        c[0].metric("Hang-off fatigue: HALYARD vs incumbent",
                    f"{cmp.hangoff_damage_reduction_frac*100:+.1f}%",
                    help="Positive = HALYARD reduces hang-off fatigue at equal grid service.")
        c[1].metric("AEP penalty (HALYARD − incumbent)",
                    f"{(cmp.halyard.aep_penalty_frac-cmp.incumbent.aep_penalty_frac)*100:+.2f}%")
        st.caption("Headline (the model chain indicates): same grid service, "
                   f"{cmp.hangoff_damage_reduction_frac*100:+.1f}% hang-off fatigue vs the "
                   "incumbent, at the stated AEP trade. No stronger claim is made.")


def tab_fatigue(p):
    st.markdown("### Fatigue detail & sensitivity")
    st.markdown(fidelity_tag("reduced") + what_this_shows(
        "Rainflow on the COMBINED stress signal (never wave-only + control-only), mean-stress "
        "correction, DNV-style S-N, Miner, and the Design Fatigue Factor. Change the S-N slope m "
        "and Goodman/Gerber in the sidebar: the number moves, the qualitative conclusion should "
        "not."), unsafe_allow_html=True)
    case = cached_case(p, True)
    hf = case.fatigue["hang_off"]
    dom = case.dominant_region()
    life = hf.life_years(p["events_per_year"], p["occurrence"])
    cols = st.columns(4)
    cols[0].metric("Hang-off damage / window", f"{hf.damage:.2e}")
    cols[1].metric("Counted cycles", f"{hf.n_cycles:.0f}")
    cols[2].metric("Dominant region", dom.replace("_", "-"))
    cols[3].metric("Hang-off fatigue life", "∞" if not np.isfinite(life) else f"{life:,.0f} yr",
                   help=f"DFF={p['DFF']:.0f}, {p['events_per_year']:.0f} events/yr, "
                        f"occurrence {p['occurrence']:.2f}.")
    a, b = st.columns(2)
    a.plotly_chart(plotting.rainflow_hist_plot(hf), use_container_width=True)
    b.plotly_chart(plotting.sn_plot(hf, make_fatigue(p).sn), use_container_width=True)
    st.plotly_chart(plotting.region_damage_bars(case), use_container_width=True)
    st.caption("Fatigue concentrates at the hang-off (stick regime, highest tension). Sag/hog/"
               "touchdown carry far less control-driven damage — the buoyancy section decouples "
               "them.")

    st.markdown("**Annualized hang-off life across the sea-state scatter**")
    st.markdown(what_this_shows(
        "Extrapolates to a lifetime over a representative 7-bin JONSWAP scatter: continuous "
        "wave fatigue by hours-in-bin, plus the control-added fatigue at the events/year rate. "
        "Counting is per-bin on the combined signal (never summed)."), unsafe_allow_html=True)
    if st.button("Compute annualized life across the scatter (14 cases)"):
        st.session_state["_run_annual"] = True
    if st.session_state.get("_run_annual"):
        from app.runner import cached_annualize
        with st.spinner("Running the sea-state scatter…"):
            ar = cached_annualize(p)
        cc = st.columns(3)
        cc[0].metric("Hang-off life (wave + control)",
                     "∞" if not np.isfinite(ar.life_years) else f"{ar.life_years:,.0f} yr")
        cc[1].metric("Hang-off life (wave only)",
                     "∞" if not np.isfinite(ar.life_years_wave_only) else
                     f"{ar.life_years_wave_only:,.0f} yr")
        share = 100 * ar.event_annual_damage / ar.total_annual_damage if ar.total_annual_damage else 0
        cc[2].metric("Control share of annual damage", f"{share:.1f}%")
        rows = [dict(Hs=b.sea.Hs_m, Tp=b.sea.Tp_s, occ=b.sea.occurrence,
                     D_wave=b.D_wave, D_support=b.D_support, added=b.added) for b in ar.bins]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def tab_ingest(p):
    st.markdown("### High-fidelity ingest (engine B)")
    st.markdown(fidelity_tag("highfi") + what_this_shows(
        "Upload precomputed OpenFAST / MoorDyn-with-EI / OrcaFlex time series and run the SAME "
        "fatigue pipeline on the real signal. Accepted columns: time, surge, pitch (platform "
        "motion) OR time, curvature_hangoff, tension_hangoff (cable response). Never blended with "
        "reduced-order numbers."), unsafe_allow_html=True)
    from app import ingest
    up = st.file_uploader("Upload CSV or parquet", type=["csv", "parquet"])
    use_example = st.checkbox("Use the bundled example dataset (synthetic stand-in)", value=up is None)
    df = None
    if up is not None:
        df = ingest.load_timeseries(up)
    elif use_example:
        try:
            df = pd.read_csv("data/ingest_example/example_platform_motion.csv", comment="#")
        except Exception:
            st.error("Example dataset not found.")
    if df is None:
        return
    try:
        res = ingest.run_ingested(df, family=get_family(), fatigue_params=make_fatigue(p),
                                  daf=p["daf"], hangoff_stick=p["hangoff_stick"])
    except ValueError as e:
        st.error(str(e))
        return
    st.success(f"Ingested {res.kind} ({len(res.t)} samples). Tagged: {res.fidelity}.")
    cols = st.columns(3)
    cols[0].metric("Hang-off damage / window", f"{res.fatigue.damage:.2e}")
    cols[1].metric("Counted cycles", f"{res.fatigue.n_cycles:.0f}")
    cols[2].metric("Columns", ", ".join(res.columns))
    st.plotly_chart(plotting.stress_plot(res.t, {"hang_off": res.stress_Pa}),
                    use_container_width=True)
    st.plotly_chart(plotting.rainflow_hist_plot(res.fatigue), use_container_width=True)


def tab_validation(p):
    st.markdown("### Validation & sensitivity")
    st.markdown(what_this_shows(
        "Does the reduced model reproduce published behaviour? Natural periods, JONSWAP Hs, cable "
        "length conservation and an analytic catenary cross-check."), unsafe_allow_html=True)
    pv = val.platform_validation()
    st.markdown("**Platform — natural periods & static lean (fitted to VolturnUS-S)**")
    st.dataframe(pd.DataFrame([
        ("Surge period (s)", f"{pv['target_periods_s']['surge']:.0f}",
         f"{pv['fitted_periods_s'][1]:.1f}"),
        ("Pitch period (s)", f"{pv['target_periods_s']['pitch']:.0f}",
         f"{pv['fitted_periods_s'][0]:.1f}"),
        ("Static surge @ rated (m)", "~20", f"{pv['static_surge_at_rated_m']:.1f}"),
        ("Static pitch @ rated (deg)", "~5.5", f"{pv['static_pitch_at_rated_deg']:.2f}"),
    ], columns=["Quantity", "Target (published)", "Fitted"]), hide_index=True,
        use_container_width=True)
    st.markdown("**Waves — JONSWAP realized Hs vs target**")
    st.dataframe(pd.DataFrame(val.wave_validation()).round(3), hide_index=True,
                 use_container_width=True)
    st.markdown("**Cable — static solve checks + analytic catenary cross-check**")
    cv = val.cable_validation()
    cb = val.catenary_benchmark()
    st.dataframe(pd.DataFrame([
        ("Length conservation (%)", round(cv["length_error_%"], 2)),
        ("Top tension (kN)", round(cv["top_tension_kN"], 1)),
        ("Max curvature (1/m)", round(cv["max_curvature_1_per_m"], 4)),
        ("MBR limit (1/m)", round(cv["mbr_limit_1_per_m"], 4)),
        ("MBR respected", cv["mbr_ok"]),
        ("Catenary κ numeric / analytic", round(cb["ratio"], 3) if cb["ratio"] == cb["ratio"] else "n/a"),
    ], columns=["Check", "Value"]), hide_index=True, use_container_width=True)
    st.markdown("**Metocean — reduced sea-state scatter (representative, disclosed)**")
    from models.metocean import SCATTER, check_occurrence_sums_to_one
    st.dataframe(pd.DataFrame([{"Hs (m)": s.Hs_m, "Tp (s)": s.Tp_s, "occurrence": s.occurrence}
                               for s in SCATTER]), hide_index=True, use_container_width=True)
    st.caption(f"Occurrences sum to {check_occurrence_sums_to_one():.2f}. The full Hs×Tp scatter "
               "is reduced to these fatigue-relevant bins (ASSUMPTIONS.md).")
    st.caption("Sensitivity: change the S-N slope m and Goodman/Gerber in the sidebar and watch "
               "the Fatigue and Comparison tabs — the qualitative conclusion survives.")
