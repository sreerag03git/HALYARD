# EXTENSIONS.md

Expert additions beyond the mandatory physics spine (§5.8), and additions **considered
and rejected** with reasons. Every implemented extension is real physics or a recognized
standard method, defaulted sensibly, toggleable, and does not change the headline claim
or the fidelity label. A rejected-with-reason list is itself expert output.

Each implemented item records: **equation/method · source · default · on/off toggle**.

---

## Implemented (behind toggles unless noted)

| # | Extension | Method / equation | Source | Default | Toggle |
|---|---|---|---|---|---|
| — | *(populated as modules land — see the per-module notes below)* | | | | |

### Aerodynamics
- **Relative-wind aero damping** — thrust uses `V_rel = V − ẋ_nacelle`, so the platform's
  own fore-aft velocity feeds back into thrust. This is *not* optional: it is the source of
  aerodynamic damping (and possible negative damping, §5.5) and is always on. Method:
  actuator-frame relative velocity in `C_T`/`C_P`. **Default: on, not toggleable** (removing
  it would falsify the stability check).

*(Turbulent inflow, blade-pitch actuator dynamics, DC-link/thermal converter limits,
multi-jurisdiction grid codes, 6-DOF hydro, frequency-dependent added mass, VIV, DFF,
seawater-CP environment factor, bilinear S-N, multi-seed UQ with confidence bands, and
Dirlik spectral cross-check are planned here — each will be added with its row above as
it is implemented.)*

---

## Considered and rejected (with reasons)

| Candidate | Why rejected (for now) |
|---|---|
| **Full EMT converter model** | Out of scope by the spec; adds no fidelity to the mechanical-fatigue chain at the timescales that matter (seconds–minutes), and would not deploy in the Streamlit envelope. The reduced-order power-command model is sufficient and the EMT boundary is stated explicitly. |
| **Two-way platform↔cable coupling** | The dynamic cable's reaction force on a 20 000-tonne platform is negligible; one-way coupling (platform → cable) is standard practice at this fidelity and keeps the live engine inside the RAM/CPU envelope. Stated in the cable module. |
| **On-the-fly OpenFAST/MoorDyn runs** | Requires Fortran compilation / licensed tools; violates the pure-pip deployment constraint. Instead, high-fidelity results are produced offline and *ingested* (engine B). |

*(This table grows as extensions are weighed.)*
