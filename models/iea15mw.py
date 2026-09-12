"""IEA Wind 15 MW reference turbine (IEA-15-240-RWT) — frozen reference values.

Type-4 (full-converter, direct-drive) offshore reference turbine. Source: Gaertner
et al., "Definition of the IEA 15-Megawatt Offshore Reference Wind Turbine",
NREL/TP-5000-75698 (2020), and the public OpenFAST model. Values below are read
into HALYARD read-only; the rotor-performance surfaces come from ``aero_surface``.

Type-4 matters (§1): the grid side is decoupled by the converter across the DC
link, so any synthetic frequency support can only be drawn from the rotor's own
kinetic energy — there is no direct electromechanical coupling to grid frequency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from models.aero_surface import AeroSurfaces, load_aero_surfaces
from physics.constants import RPM_TO_RADS


@dataclass(frozen=True)
class Turbine:
    # Rating / rotor geometry --------------------------------------------------
    rated_power_W: float = 15.0e6          # rated electrical power [W]
    rotor_radius_m: float = 120.0          # rotor radius R [m] (D = 240 m)
    hub_height_m: float = 150.0            # hub height above MSL [m] (thrust lever arm)

    # Rotor speed envelope -----------------------------------------------------
    omega_rated_rads: float = 7.56 * RPM_TO_RADS   # rated rotor speed [rad/s] (0.7917)
    omega_min_rads: float = 5.00 * RPM_TO_RADS     # min operating speed [rad/s] (0.5236)
    omega_max_rads: float = 7.56 * RPM_TO_RADS     # max (rated) rotor speed [rad/s]

    # Inertia ------------------------------------------------------------------
    # Rotor + drivetrain inertia about the low-speed shaft [kg m^2].
    # NREL/TP-5000-75698 rotor inertia ~3.1557e8; direct-drive generator inertia is
    # small and folded in here. Recorded in ASSUMPTIONS.md.
    rotor_inertia_kgm2: float = 3.1557e8

    # Operating envelope -------------------------------------------------------
    rated_wind_ms: float = 10.59           # rated wind speed [m/s]
    cut_in_wind_ms: float = 3.0
    cut_out_wind_ms: float = 25.0
    converter_efficiency: float = 0.95     # eta_conv (drivetrain+converter) [-]

    # Rotor-performance surfaces (loaded lazily; NREL deck if present else reduced)
    aero: AeroSurfaces = field(default_factory=load_aero_surfaces)

    @property
    def rotor_area_m2(self) -> float:
        import math
        return math.pi * self.rotor_radius_m ** 2

    @property
    def base_inertia_constant_s(self) -> float:
        """Physical inertia constant H = 0.5 J omega_rated^2 / S_base [s].

        This is the *actual* kinetic-energy store of the rotor, distinct from the
        emulated inertia constant H_wt used in the synthetic-inertia control law.
        """
        return 0.5 * self.rotor_inertia_kgm2 * self.omega_rated_rads ** 2 / self.rated_power_W

    def kinetic_energy_J(self, omega_rads: float) -> float:
        return 0.5 * self.rotor_inertia_kgm2 * omega_rads ** 2

    def available_energy_J(self, omega_rads: float) -> float:
        """Extractable KE down to the minimum rotor speed [J] (>= 0)."""
        e = 0.5 * self.rotor_inertia_kgm2 * (omega_rads ** 2 - self.omega_min_rads ** 2)
        return max(e, 0.0)

    def tip_speed_ratio(self, omega_rads: float, wind_ms: float) -> float:
        return omega_rads * self.rotor_radius_m / max(wind_ms, 1e-6)


# Frozen singleton — import this everywhere the turbine is needed.
IEA15MW = Turbine()
