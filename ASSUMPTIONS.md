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

*(To be finalized in the cable module; every geometric and material number will be
listed here with DNV-RP-F401 / DNVGL-RP-0360 rationale and flagged **[representative]**.)*

---

## 4. Grid & metocean

*(Swing-equation inertia constants, load-damping, and JONSWAP sea-state bins will be
listed here as their modules are built; the bundled default frequency trace and its
provenance will be recorded.)*

---

## 5. Fatigue S-N / T-N curve

*(S-N slope m, log C, knee, σ_u and the curve's source — DNV-RP-C203 seawater-with-CP
class, representative of the armour-wire steel — will be listed here when the fatigue
module is built, with the simplified equivalent-stress assumption for helical armour
flagged **[representative]**.)*
