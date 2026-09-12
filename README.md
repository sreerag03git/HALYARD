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
| **A — live reduced-order** | The physics in `physics/`, solved in real time inside Streamlit (swing-equation grid, synthetic-inertia/droop control, thrust–λ coupling, 2-DOF platform, lazy-wave cable, rainflow fatigue). | `reduced-order (live)` |
| **B — ingested high-fidelity** | Precomputed OpenFAST / MoorDyn-with-EI / OrcaFlex time series uploaded as files, run through the same fatigue pipeline. | `high-fidelity (ingested)` |

Reduced-order output is *never* presented as high-fidelity.

## Reference models (all public, frozen, read-only)

- **Turbine:** IEA Wind 15 MW reference turbine (IEA-15-240-RWT), NREL/TP-5000-75698. Type-4 direct-drive.
- **Platform:** UMaine VolturnUS-S semi-submersible, NREL/TP-5000-76773.
- **Dynamic cable:** a representative lazy-wave lay assembled to DNV-RP-F401 / DNVGL-RP-0360 geometry rules (documented in `ASSUMPTIONS.md`).
- **Grid frequency:** synthetic events from a swing-equation model, plus upload of measured low-inertia traces.

See `ASSUMPTIONS.md` for every representative value and its source, and `EXTENSIONS.md` for expert additions beyond the mandatory spine.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

Point Streamlit Community Cloud at this repo; the entry point is `streamlit_app.py` at the repo root. `requirements.txt` is pinned and pure-pip (no OpenFAST/Fortran/OrcaFlex compilation). The app is designed to run within the ~1 GB / single-process envelope: long runs are cached (`st.cache_data`) and seeded for reproducibility.

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
