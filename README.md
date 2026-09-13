# HALYARD

**Quantifying the cable-fatigue cost of grid-frequency support on a floating offshore wind turbine — and shaping the recovery to manage it.**

HALYARD is a physics-based simulation-and-analysis application. It follows one continuous, explicit chain of causation:

> grid frequency → converter power command → generator torque → rotor speed → aerodynamic thrust → floating-platform motion → dynamic-cable curvature & tension → hang-off stress → fatigue

and asks a single, narrowly defensible question:

> *Same grid service, measurably different hang-off fatigue, at a quantified energy-capture trade-off?*

Every number the app displays is computed from a running physical model or read from a real dataset. When the effect is not real under the chosen conditions, the app says so plainly (the **Phase-1 honesty gate**) and does not proceed to claim a saving. All results are stated as *"the model chain indicates,"* with uncertainty.

---

## The concept in one paragraph

A full-converter (Type-4) floating turbine is controlled electrically but floats mechanically. When the converter modifies power to support grid frequency, it can only draw on the rotor's kinetic energy (the DC link decouples the grid side). Rotor speed changes → tip-speed ratio changes → thrust coefficient changes → the horizontal rotor force that pushes the platform changes → the platform drifts and pitches slowly → the dynamic power cable flexes at its hang-off and fatigues. Standard cable-fatigue calculations assume the cable is moved only by waves; the motion added by frequency-support control is not in that picture. The grid-mandated **support** phase cannot be reshaped, but the **recovery** phase (spinning the rotor back up) is a free design choice. HALYARD shapes the recovery to cut the slow platform excursion that drives hang-off fatigue, while still meeting every grid-code requirement — and reports the AEP trade honestly.

## The fidelity ladder

HALYARD is explicit about where every number comes from. Two engines exist and are never blended silently:

| Engine | What it is | Label on every figure |
|---|---|---|
| **A — live** | The physics in `physics/`, solved in real time inside Streamlit. | `reduced-order (live)` |
| **B — ingested high-fidelity** | Precomputed OpenFAST / MoorDyn-with-EI / OrcaFlex time series uploaded as files, run through the same fatigue pipeline. | `high-fidelity (ingested)` |

The live engine has been developed toward industry-grade physics:

- **Grid** — swing-equation SFR with a primary governor response and a filtered-RoCoF (PLL) state.
- **Control** — synthetic-inertia + droop support and a shaped recovery lever; a **ROSCO-style blade-pitch controller** (with floating feedback) that extends operation above rated.
- **Aerodynamics** — the **real IEA-15MW OpenFAST rotor deck** (Cp/Ct over pitch×TSR), relative-wind aero damping, a dynamic-inflow lag, and optional **Kaimal turbulent inflow**.
- **Platform** — a **6-DOF potential-flow model**: added mass / radiation damping / excitation RAOs from a **boundary-element (Capytaine) solve** of the VolturnUS-S hull, run in the time domain via the **Cummins equation** with radiation memory; natural periods validated against the published values. (A fast fitted 2-DOF screening engine is also selectable.)
- **Cable** — a quasi-static lazy-wave family *and* a **dynamic lumped-mass FE cable** (Morison hydro, bending, seabed friction, **VIV** wake oscillator) that computes the dynamic amplification self-consistently.
- **Fatigue** — rainflow → mean-stress → S-N → Miner on the combined signal, with a **DLC matrix** (wind × sea-state × heading × seed) and **multi-seed + S-N-scatter confidence bands**.

The BEM hydrodynamic database and the rotor deck are computed/collected **offline** and bundled; the live engine reads them (no OpenFAST/OrcaFlex/BEM at runtime). Reduced-order output is *never* presented as high-fidelity.

## Reference models (all public, frozen, read-only)

- **Turbine:** IEA Wind 15 MW reference turbine (IEA-15-240-RWT), NREL/TP-5000-75698. Type-4 direct-drive.
- **Platform:** UMaine VolturnUS-S semi-submersible, NREL/TP-5000-76773.
- **Dynamic cable:** a representative lazy-wave lay assembled to DNV-RP-F401 / DNVGL-RP-0360 geometry rules (documented in `ASSUMPTIONS.md`).
- **Grid frequency:** synthetic events from a swing-equation model, plus upload of measured low-inertia traces.

See `ASSUMPTIONS.md` for every representative value and its source, and `EXTENSIONS.md` for expert additions beyond the mandatory spine.

## The analysis flow

1. **Overview** — the system (interactive 3-D model from real geometry) and the causation chain.
2. **Grid & control / Platform & cable** — the coupled event, the platform motion, the cable stress.
3. **Phase-1 gate** — the honesty check: does control add meaningful hang-off fatigue vs waves, against *pre-registered* thresholds? Returns **PASS** or **NEGLIGIBLE**.
4. **Recovery sweep (Stage A)** and **Comparison (Stage B)** — unlocked only on PASS. Stage B first proves the incumbent and HALYARD recoveries are *grid-code equivalent*, then compares hang-off fatigue and the AEP trade.
5. **Fatigue detail** — rainflow → mean-stress → S-N → Miner on the combined signal, plus annualized life across a 7-bin sea-state scatter; live sensitivity to the S-N slope and Goodman/Gerber.
6. **High-fidelity ingest** and **Validation**.

A key honest finding the tool surfaces: control adds a meaningful fatigue fraction **within calm, frequent seas**, but integrated over the full scatter **storm waves dominate annual hang-off fatigue**, so the annual share is small. The headline is deliberately narrow.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

Point Streamlit Community Cloud at this repo (a private repo works — link your GitHub); the entry point is `streamlit_app.py` at the repo root. `requirements.txt` is pinned and pure-pip (no OpenFAST/Fortran/OrcaFlex compilation). The app runs within the ~1 GB / single-process envelope: the lazy-wave cable family and every simulation are cached (`st.cache_resource` / `st.cache_data`) and seeded for reproducibility.

**First load computes** the cable family and the Phase-1 gate (tens of seconds, with a spinner); afterwards everything is cached and instant until inputs change. The multi-run stages (recovery sweep, three-controller comparison, across-scatter annualization) are behind explicit **Run** buttons.

Continuous integration (`.github/workflows/ci.yml`) runs the pytest suite and a headless smoke test of the app on every push.

## Cited choices

- **Grid-code jurisdiction:** GB / ENTSO-E Continental Europe (50 Hz). KPIs: RoCoF containment, frequency nadir, response energy in the mandated window, no sustained secondary violation.
- **Incumbent recovery baseline:** a cited fast-frequency-response recovery from the literature (see `ASSUMPTIONS.md`) — never a strawman.

## What this is not

Not a mock-up, not a slide, not a preset animation. No AI-generated or stock imagery — every visual is drawn from computed geometry or real data. No fabricated constants. No claim stronger than *"same grid service, different hang-off fatigue, at a quantified AEP trade."* Electromagnetic-transient (EMT) converter modelling is explicitly out of scope; the grid/converter representation is reduced-order.

## Repository layout

```
models/     frozen reference decks + values (turbine, platform, cable, aero surfaces)
physics/    solvers: grid, controller, aero coupling, platform, cable, fatigue, simulation
analysis/   phase-1 gate, recovery sweep, three-controller comparison, validation
app/        Streamlit UI, plotting, 3-D system model
configs/    one YAML fully defines a case (incl. committed Phase-1 thresholds)
data/       bundled frequency traces + example ingest dataset
tests/      pytest suite (physics vs analytical/reference cases; Phase-1 FAIL guaranteed)
```

## Status

Under active construction following the build order in the specification (§14). See commit history.

## License / provenance

Reference values are from public sources cited above. All results are *"the model chain indicates."*
