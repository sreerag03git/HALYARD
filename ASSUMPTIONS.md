# ASSUMPTIONS.md

Every representative assumption in HALYARD, with its source or explicit rationale.
A value flagged **[published]** is read from a cited public reference; **[representative]**
is a value HALYARD assembled (with the reasoning given); **[fitted]** is calibrated to
reproduce a published behaviour, with the fit target stated.

All results in the app are *"the model chain indicates,"* with uncertainty.

---

## 1. Turbine — IEA-15-240-RWT (`models/iea15mw.py`)

Source: Gaertner et al., *Definition of the IEA 15-Megawatt Offshore Reference Wind
Turbine*, NREL/TP-5000-75698 (2020), and the public OpenFAST model.

| Quantity | Value | Basis |
|---|---|---|
| Rated power | 15 MW | [published] |
| Rotor radius / diameter | 120 m / 240 m | [published] |
| Hub height | 150 m | [published] |
| Rated rotor speed | 7.56 rpm (0.7917 rad/s) | [published] |
| Minimum rotor speed | 5.0 rpm (0.5236 rad/s) | [published] |
| Rated wind speed | 10.59 m/s | [published] |
| Rotor+drivetrain inertia (about LSS) | 3.1557×10⁸ kg·m² | [published] rotor inertia; direct-drive generator inertia is comparatively small and folded in here **[representative]** |
| Converter+drivetrain efficiency | 0.95 | [representative] typical full-converter value |

### 1a. Rotor performance surfaces C_P(λ,β), C_T(λ,β) — **important provenance**

The authoritative source is the public OpenFAST rotor-performance deck `Cp_Ct_Cq.txt`.
**If that file is placed at `models/data/Cp_Ct_Cq.txt`, HALYARD loads it verbatim** and
labels the surfaces high-fidelity.

When it is absent (clean clone), HALYARD builds **[representative]** reduced-order
analytical surfaces and labels them as such everywhere (`AeroSurfaces.provenance`):
- C_P: empirical Heier/Slootweg C_P(λ,β) form, affine-calibrated in tip-speed-ratio
  and amplitude to the published anchors **C_P,max = 0.489 at λ_opt = 9.0, β = 0**.
- C_T: a shaped surface (rises ~linearly with λ, peaks at λ≈10.2, decays) calibrated
  so the region-2 operating point gives **C_T ≈ 0.79**, i.e. **~2.4 MN rated thrust**
  at rated wind — the published-order rotor thrust. Verified in the build/verification
  step (thrust = 2.45 MN, C_P,max = 0.489 at λ = 9.03).
- These are **not** the NREL deck. They are calibrated to real anchor points using a
  recognized empirical form, so the *local* dC_T/dλ near the operating point (the
  mechanism that turns rotor-speed changes into thrust transients) is physically
  reasonable (positive through 7 < λ < 10).

---

## 2. Platform — UMaine VolturnUS-S (`models/volturnus_s.py`)

Source: Allen et al., *Definition of the UMaine VolturnUS-S Reference Platform…*,
NREL/TP-5000-76773 (2020).

| Quantity | Value | Basis |
|---|---|---|
| Water depth (reference site) | 200 m | [published] |
| Draft / freeboard | 20 m / 15 m | [published] |
| Column diameter | 12.5 m | [published] |
| Column spacing (centre→offset) | 51.75 m | [published] |
| Displacement | 20 206 m³ | [published] |
| Platform + ballast mass | 17.839×10⁶ kg | [published] |
| Tower mass | 1.263×10⁶ kg | [published] |
| RNA mass | 1.017×10⁶ kg | [published] |
| Fairlead depth | 14 m below MSL | [representative] typical for this hull |
| System COG height | −1.0 m (near MSL) | [fitted/derived] mass-weighted from components |

### 2a. Reduced-order 2-DOF (surge, pitch) motion model — **the fit, stated**

The spec permits fitting reduced coefficients to published dynamics. HALYARD anchors
the two quantities that govern the fatigue mechanism and verifies the result:

1. **Static lean per unit thrust** *(decisive for slow-drift mean-stress shift)*:
   surge stiffness set so rated thrust (2.4 MN) → **~20 m mean surge** **[representative,
   published-order]**; pitch stiffness so it → **~5.5° mean pitch** **[representative,
   published-order]**. Verified fit gives 21.5 m and 6.05° (coupling shifts them slightly).
2. **Natural periods** *(resonant amplification)*: effective (rigid+added) inertias
   fitted so **surge period = 120 s, pitch period = 28 s** **[fitted to VolturnUS-S
   literature]**. Verified fit gives 120.4 s and 28.0 s.
3. **Damping**: modal damping ratios **ζ_surge = 0.05, ζ_pitch = 0.03** **[representative]**
   (radiation + viscous + mooring), applied in the modal basis.

Mooring surge–pitch coupling is included via the horizontal mooring load acting at
fairlead depth. The fit is surfaced in the app's **Validation** tab (§10).

---

## 3. Dynamic cable — representative lazy-wave lay (`models/dynamic_cable.py`)

All values **[representative]**, internally consistent with a 66 kV three-core dynamic
cable and laid out to DNV-RP-F401 / DNVGL-RP-0360 lazy-wave geometry rules. The solved
static shape (verified) is a proper lazy wave: hang-off at −8 m, upper sag bend, a
buoyancy hog crest at ≈ −87 m, touchdown at ≈ 134 m horizontal, then seabed to the anchor.

| Quantity | Value | Basis |
|---|---|---|
| Water depth | 200 m | matches VolturnUS-S site |
| Outer diameter | 0.20 m | typical 66 kV 3-core dynamic |
| Dry mass / length | 45 kg/m | → submerged weight ≈ 125 N/m |
| Axial stiffness EA | 400 MN | typical dynamic-cable value |
| Bending stiffness EI | 25 kN·m² | flexible dynamic cable |
| Operational MBR | 3.5 m | curvature limit 0.286 1/m |
| Total suspended length | 320 m | gives lazy-wave slack |
| Hang-off → anchor horizontal | 180 m | design layout |
| Buoyancy section | 42–62 % of length | creates the hog |
| Buoyancy net uplift | 1.6 × bare submerged wt | gentle hog, no surface breach |

**Equivalent-stress model [representative]** — `σ = E_steel·r_bend·κ + T/A_armour`:
- Sag/hog/touchdown use the **full-slip** wire radius (2.5 mm), consistent with the low EI
  (armour slips) — physically bounded stresses.
- The **hang-off** uses a **calibrated effective (partial-slip) lever** `hangoff_bend_radius`
  = **18 mm** as the DEFAULT. It is bracketed by the full-slip wire radius (2.5 mm, lower)
  and the no-slip armour pitch radius (90 mm, stick upper bound). It is set so the reference
  lay + bend stiffener achieves a **conventional dynamic-cable design life** over the sea-state
  scatter (≈ 250 yr calculated, DFF 3) — standard design practice. **The RELATIVE control
  effect (Phase-1 added fraction, Stage-B reduction) is robust to this lever**; the absolute
  life is a rough indicator. `hangoff_stick=False` switches to the slip lower bound (sensitivity).
- Solve method: near-inextensible Position-Based Dynamics + Gauss-Seidel length polish
  (≈ 0.07 % suspended-length error); a light Laplacian bending term and seabed-tail
  straightening remove discretization buckling; tension from force balance.
- Dynamic response: quasi-static family (re-solve vs hang-off offset) + DAF (default 1.2),
  with the hang-off bend-stiffener curvature driven by platform pitch (stiffener length 4 m).

---

## 4. Grid & metocean

**Swing-equation SFR grid (`physics/grid.py`)** — all defaults **[representative]** of a
low-inertia system; the effect scales with grid weakness so this is the relevant regime:

| Quantity | Value | Basis |
|---|---|---|
| Nominal frequency | 50 Hz | GB / ENTSO-E CE jurisdiction |
| System inertia H_sys | 2.5 s (low-inertia default; 1.5–6 range) | [representative] |
| Load damping D | 1.0 pu/pu (~2 %/Hz) | [representative] |
| Rest-of-system droop R_sys | 0.05 | [representative] primary response |
| Primary-response lag T_gov | 8 s | [representative] aggregate governor+prime-mover |
| Spinning reserve | 0.12 pu | [representative] |
| RoCoF measurement filter | 0.5 s | [representative] PLL/measurement lag |
| System base | 3000 MW | [representative] island/region |
| Supporting wind capacity | 900 MW (participation 0.30) | [representative] wind-rich low-inertia grid |

**Controller** — support: synthetic inertia H_wt (control gain, default 6 s), droop
R = 0.05, deadband 15 mHz, magnitude clip 0.10 pu, rate limit, reservoir-guarded at ω_min.
Recovery lever: τ_rec, rate_rec, shape exponent (§5.3). Incumbent baseline recovery is a
fast first-order return (τ ≈ 2 s) representative of published FFR schemes — **not** a strawman.

**Metocean scatter [representative]** (`models/metocean.py`) — the full Hs×Tp scatter is
reduced to a disclosed **7-bin** set, weighted toward frequent calm/moderate seas (a
North-Sea-like distribution), occurrences summing to 1.0:

| Hs (m) | 0.75 | 1.25 | 1.75 | 2.5 | 3.5 | 4.5 | 6.0 |
|---|---|---|---|---|---|---|---|
| Tp (s) | 6.5 | 7.5 | 8.5 | 9.5 | 10.5 | 11.5 | 12.5 |
| occ. | 0.22 | 0.24 | 0.20 | 0.16 | 0.10 | 0.06 | 0.02 |

Annualization (`analysis/annualize.py`): continuous wave fatigue scaled by hours-in-bin +
control-added fatigue at the events/year rate, per bin on the combined signal. A finding:
**over the full scatter, storm bins dominate annual hang-off fatigue, so the control's
annual share is small even though it adds a meaningful fraction within calm, frequent seas.**

---

## 5. Fatigue S-N / T-N curve (`physics/fatigue.py`)

| Quantity | Value | Basis |
|---|---|---|
| S-N form | bilinear, MPa-based | DNV-RP-C203-style steel curve **[representative]** of armour-wire steel in seawater with cathodic protection |
| m₁, log a₁ (N < 10⁶) | 3.0, 11.764 | DNV-RP-C203 D-curve-style |
| m₂, log a₂ (N > 10⁶) | 5.0, 15.606 | second slope past the knee |
| UTS σ_u (mean-stress) | 1400 MPa | armour-wire steel [representative] |
| Mean-stress correction | Goodman (default) / Gerber (toggle) | §5.7 |
| Design Fatigue Factor | 3 (range 3–10 for critical/inaccessible) | DNV §5.8 |

Fatigue is counted on the **combined** stress signal (never wave-only + control-only), so
the mean-stress shift from the slow control drift is captured (R-ratio effect). The
helical-armour equivalent-stress simplification is flagged in §3.
