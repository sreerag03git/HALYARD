"""Named, sourced physical constants for HALYARD.

Every constant here is either a universal physical constant or a value traced to a
public reference (see §4 of the specification and ASSUMPTIONS.md). No magic numbers
live inside the solver code — they are named here or supplied through a case YAML.

Symbols carry SI units in their docstring. Sources are cited inline; representative
assumptions (values we assembled rather than read from a reference) are flagged
``REPRESENTATIVE`` and recorded in ASSUMPTIONS.md.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Universal constants
# ---------------------------------------------------------------------------
G = 9.80665           # gravitational acceleration [m/s^2]
RHO_WATER = 1025.0    # seawater density [kg/m^3]
RHO_AIR = 1.225       # air density at sea level [kg/m^3]

# ---------------------------------------------------------------------------
# Grid / jurisdiction (GB ESO / ENTSO-E Continental Europe use 50 Hz)
# ---------------------------------------------------------------------------
F0_HZ = 50.0          # nominal system frequency [Hz]

# ---------------------------------------------------------------------------
# Unit conversions used across the codebase
# ---------------------------------------------------------------------------
RPM_TO_RADS = 2.0 * 3.141592653589793 / 60.0   # rev/min -> rad/s
PA_TO_MPA = 1.0e-6
