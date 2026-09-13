"""Representative lazy-wave dynamic power cable — frozen reference lay (§4, §5.6).

This is the one input HALYARD assembles rather than reads from a single turbine-specific
reference. Every geometric and material number is **[representative]**, chosen to be
internally consistent with a 66 kV three-core dynamic cable and laid out to DNV-RP-F401 /
DNVGL-RP-0360 lazy-wave geometry rules (a buoyancy section creates a hog/sag arch that
decouples platform motion from the seabed touchdown). All values are recorded in
ASSUMPTIONS.md; a fatigue number computed on an undocumented lay is worthless.

Equivalent-stress model (simplified, explicitly stated)
-------------------------------------------------------
Real cables carry helical armour whose true stress state (inter-wire friction, slip,
load sharing) is complex. HALYARD uses a simplified equivalent-stress model at the
fidelity of this study (§5.6):
    sigma(t) = E_steel * r_bend * kappa(t)  +  T(t) / A_armour
where r_bend is a bending lever arm and A_armour the total armour steel area.

Choice of r_bend (slip vs no-slip) — this matters:
* Default **slip regime** uses the individual armour-wire radius (r_wire ~ 2.5 mm). This
  is consistent with the cable's LOW bending stiffness (EI = 25 kN m^2 is ~100x below the
  fully-bonded E_steel*I_armour), which proves the armour wires slip; the cyclic stress is
  then dominated by each wire bending about its own axis, ~ E_steel * r_wire * kappa. This
  yields physically bounded stresses (a fully-bonded R_armour lever would predict >UTS
  stresses at the hog and is unphysical for a flexible dynamic cable).
* The **no-slip / stick** bound uses the armour pitch radius R_armour and is offered as a
  higher-fidelity toggle for the hang-off (where high tension can cause stick, the reason
  hang-offs are fatigue-critical). See EXTENSIONS.md.

Assumptions: full tension sharing by the armour, single equivalent wire radius, no explicit
inter-wire friction model. Flagged in ASSUMPTIONS.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from physics.constants import G, RHO_WATER


@dataclass(frozen=True)
class DynamicCable:
    # Environment ------------------------------------------------------------
    water_depth_m: float = 200.0

    # Cross-section / material (66 kV 3-core dynamic cable, representative) ----
    outer_diameter_m: float = 0.20
    dry_mass_per_len_kgm: float = 45.0        # mass in air [kg/m]
    EA_N: float = 400.0e6                     # axial stiffness [N]
    EI_Nm2: float = 25.0e3                    # bending stiffness [N m^2]
    min_bend_radius_m: float = 3.5            # operational MBR [m] (curvature limit 1/MBR)

    # Equivalent-stress model (armour) ---------------------------------------
    E_steel_Pa: float = 200.0e9               # armour steel Young's modulus [Pa]
    armour_pitch_radius_m: float = 0.090      # radius of the armour layer [m] (no-slip bound)
    armour_wire_radius_m: float = 0.0025      # individual armour-wire radius [m] (slip regime)
    armour_area_m2: float = 1.5e-3            # total armour steel area [m^2]

    # Lazy-wave layout (representative; solved shape verified in the cable module) --
    hangoff_x_m: float = 0.0                  # hang-off horizontal position [m]
    hangoff_z_m: float = -8.0                 # hang-off depth below MSL [m]
    total_length_m: float = 310.0             # total cable length [m] (tuned: clean lazy wave)
    horizontal_layout_m: float = 165.0        # hang-off -> subsea anchor horizontal [m]
    buoyancy_start_frac: float = 0.42         # buoyancy section start (fraction of length)
    buoyancy_end_frac: float = 0.62           # buoyancy section end (fraction of length)
    buoyancy_uplift_factor: float = 1.6       # net uplift as a multiple of bare submerged wt

    @property
    def buoyancy_from_water_N_per_m(self) -> float:
        return RHO_WATER * (math.pi / 4.0 * self.outer_diameter_m ** 2) * G

    @property
    def submerged_weight_N_per_m(self) -> float:
        """Net submerged weight of the bare cable [N/m] (positive downward)."""
        dry_w = self.dry_mass_per_len_kgm * G
        return dry_w - self.buoyancy_from_water_N_per_m

    @property
    def buoyancy_section_net_N_per_m(self) -> float:
        """Net vertical load per length in the buoyancy section [N/m] (negative = up).

        The buoyancy modules provide net uplift of ``buoyancy_uplift_factor`` times the
        bare cable's submerged weight, so the section arches upward into a hog.
        """
        return -self.buoyancy_uplift_factor * self.submerged_weight_N_per_m

    @property
    def curvature_limit_1_per_m(self) -> float:
        return 1.0 / self.min_bend_radius_m


REFERENCE_CABLE = DynamicCable()
