# EXTENSIONS.md

Expert additions beyond the mandatory physics spine (§5.8), and additions **considered and
rejected** with reasons. Every implemented extension is real physics or a recognized
standard method, defaulted sensibly, toggleable where it affects the result, deployable
within the Streamlit envelope, and does not silently change the headline claim or the
fidelity label. The rejected-with-reason list is itself expert output.

Each implemented item records **method/equation · source · default · toggle**.

---

## Implemented

| Extension | Method / equation | Source | Default | Toggle |
|---|---|---|---|---|
| **Relative-wind aero damping** | Thrust & torque use `V_rel = V − ẋ_nacelle`, `ẋ_nacelle = ẋ_surge + z_hub·θ̇`. This is the source of aerodynamic damping (and possible negative damping). | Standard actuator-frame reduced-order aero | **on** | not toggleable (removing it would falsify the stability check) |
| **Filtered RoCoF measurement state** | `T_roc·d(RoCoF_meas)/dt = RoCoF_true − RoCoF_meas`; synthetic inertia acts on the filtered RoCoF. Breaks the algebraic loop and mirrors real PLL/measurement lag. | §5.8 grid item; PLL practice | **on** (T_roc = 0.5 s) | via `T_rocof_s` in config |
| **Primary governor response in the SFR** | First-order reserve-limited primary response `T_gov·dP_gov/dt = −P_gov − (1/R)Δf`, so frequency arrests at a realistic quasi-steady offset instead of running away on load-damping alone. | Anderson–Mirheydar SFR reduced model | **on** | via `R_sys`, `T_gov_s`, `reserve_pu` |
| **KE-reservoir guard** | Support command truncated when `ω ≤ ω_min + margin` — an undeliverable command is truncated, not honoured. | §5.2 hard constraint | **on** | — |
| **Bend-stiffener hang-off model** | Hang-off curvature driven by the platform PITCH rotating the stiffener relative to the cable far field: `κ = [Δα_natural − Δθ]/L_stiffener` (+ static). | Standard hang-off/bend-stiffener fatigue reasoning | **on** | `L_stiffener` constant |
| **Armour stick/slip bending regime** | Bending lever = armour pitch radius (stick, high-tension hang-off) or wire radius (slip). A partial helical-armour stress refinement. | §5.8 helical-armour item; DNV cable practice | hang-off **stick** | `hangoff_stick` checkbox (sensitivity) |
| **Bilinear S-N with knee** | Two-slope `log N = log a − m·log S_eq`, knee at N_knee. | DNV-RP-C203 style | **on** | `m1, m2, log_a, N_knee` |
| **Mean-stress correction** | Goodman (primary) / Gerber (sensitivity) / none, on the **combined** signal. | §5.7 | **Goodman** | sidebar selector |
| **Design Fatigue Factor** | Allowable life divided by DFF (3–10 for critical/inaccessible). | DNV §5.8 | **3** (5 in low_inertia_severe) | `DFF` slider |
| **Dynamic amplification factor** | DAF on the fluctuating part of curvature/tension (the §5.6 option-a dynamic correction). | §5.6 | **1.2** | `daf` slider |
| **Pitch negative-damping check** | Effective pitch damping incl. aero feedback `B55 − (dF/dẋ)·z_hub²`; cases with ζ ≤ 0 are flagged unstable and excluded from the sweep. | §5.5 real failure mode | **on** | — |
| **Grid-code compliance framing** | Equivalence = both controllers meet the code (RoCoF/nadir/secondary/energy limits), not identical KPIs. Multiple KPIs computed. | §7 grid codes | **on** | limits in `GridCodeLimits` |
| **Config-driven cases + committed thresholds** | One YAML fully defines a case incl. pre-registered Phase-1 thresholds. | §11 | **on** | per-case YAML |
| **High-fidelity ingest (engine B)** | Same fatigue pipeline on uploaded OpenFAST/MoorDyn/OrcaFlex series; tagged separately, never blended. | §9 | — | ingest tab |
| **6-DOF BEM hydrodynamics** (major) | Potential-flow BEM (Capytaine) → A(ω), B(ω), excitation RAOs, computed offline and bundled; Cummins time-domain with radiation-memory convolution + infinite-frequency added mass; analytic hydrostatics; mooring fitted to the published 6-DOF natural periods (validated). Replaces the fitted 2-DOF as the default. | Cummins (1962); potential-flow hydro | **on (6-DOF default)** | sidebar engine radio (2-DOF fast screening) |
| **Wind-wave misalignment** | Wave heading fed to the BEM excitation RAOs (0–90°), exciting sway/roll/yaw off head seas. | §5.8 hydro | 0° (aligned) | sidebar heading slider |

---

## Considered and rejected (with reasons)

| Candidate | Why rejected / deferred |
|---|---|
| **Full EMT converter model** | Out of scope by the spec; adds no fidelity to the mechanical-fatigue chain at the seconds–minutes timescales that matter, and would not deploy in the Streamlit envelope. The reduced-order power-command model is sufficient and the EMT boundary is stated explicitly. |
| **Two-way platform ↔ cable coupling** | The dynamic cable's reaction on a ~20 000 t platform is negligible; one-way coupling is standard at this fidelity and keeps the live engine inside the CPU/RAM envelope. |
| **On-the-fly OpenFAST / MoorDyn runs** | Requires Fortran compilation / licensed tools; violates the pure-pip deployment constraint. High-fidelity results are produced offline and *ingested* (engine B). |
| **Turbulent inflow (Kaimal/von-Kármán)** | Deferred: steady wind isolates the control-driven effect from turbulence noise. Turbulence would require multi-seed averaging (a UQ extension) to see the control signal, multiplying run cost beyond the live envelope. The clean control-vs-wave comparison is more honest at this fidelity. |
| ~~6-DOF platform~~ | **NOW IMPLEMENTED** (see the implemented table) — full 6-DOF BEM Cummins hydro is the default engine. The quasi-static lazy-wave shows slow heave is geometrically decoupled from hang-off tension (~1 N/m); the *dynamic* heave-induced tension is deferred to the dynamic FE cable upgrade. |
| **Blade-pitch control above rated** | Deferred: scenarios default to wind ≤ rated (blade pitch = 0), where the rotor has both KE and aero headroom for support and the aero surfaces are used in their validated region. Above-rated operation is guarded against in the UI. |
| **Vortex-induced vibration (VIV) budget** | Deferred: VIV is a separate high-frequency mechanism needing its own reduced-order model; it would add to the *total* fatigue budget (making the control share smaller), so omitting it is non-conservative for the headline but keeps scope focused on the control mechanism. Flagged as the main missing budget item. |
| **Multi-seed UQ with confidence bands** | Deferred to a documented single seed (exposed in the UI). Running many seeds per case to produce a band is a clear next step but multiplies the live run cost; the seed is surfaced so a user can sample it manually. |
| **Dirlik / spectral fatigue cross-check** | Deferred: a useful cross-check on rainflow, but rainflow on the combined signal is the correct primary method for the mean-shift (R-ratio) mechanism, which a narrow-band spectral method would miss. |
| **Frequency-dependent added mass (retardation/convolution)** | Deferred: constant added-mass/damping fitted to the published natural periods is adequate for the low-frequency drift and the near-resonant pitch response that drive the result; full retardation is offline-model territory. |
| **Second-order slow-drift wave forces** | Deferred: these would add low-frequency wave excursion that competes with the control drift. Omitting them is conservative for attributing the slow drift to control, and is stated as a limitation. |
