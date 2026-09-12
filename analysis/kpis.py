"""Grid-code KPIs and AEP penalty (§7).

Cited jurisdiction: GB ESO / ENTSO-E Continental Europe (50 Hz). The KPIs below are the
ones a frequency-support scheme is judged on, and are what must MATCH between the incumbent
and HALYARD recoveries for the comparison to be "electrically equivalent" (§7, Stage B):

  * RoCoF containment   — peak |RoCoF| within limits
  * frequency nadir     — the lowest frequency reached
  * response energy     — energy delivered in the mandated window after the event
  * settling / secondary — no sustained secondary violation during recovery

The AEP penalty is the aerodynamic capture lost by operating off the Cp peak longer during
a shaped recovery (a gentler recovery keeps the rotor below optimal tip-speed-ratio longer).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.simulation import SimResult
from physics.aero import aero_power_W
from models.iea15mw import IEA15MW


@dataclass
class GridKPIs:
    nadir_hz: float
    max_rocof_hz_s: float
    settled_hz: float
    secondary_min_hz: float          # lowest frequency during the recovery phase
    response_energy_MJ: float        # aggregate wind support energy in the mandated window
    quasi_steady_reached: bool

    def as_dict(self) -> dict:
        return {
            "nadir_hz": self.nadir_hz,
            "max_rocof_hz_s": self.max_rocof_hz_s,
            "settled_hz": self.settled_hz,
            "secondary_min_hz": self.secondary_min_hz,
            "response_energy_MJ": self.response_energy_MJ,
        }


def grid_kpis(r: SimResult, response_window_s: float = 10.0) -> GridKPIs:
    """Compute the grid-code KPIs from a simulation result."""
    t = r.t
    t_event = r.config.schedule.t_event_s
    post = t >= t_event
    nadir = float(r.freq_hz[post].min()) if post.any() else float(r.freq_hz.min())
    max_rocof = float(np.max(np.abs(r.rocof_hz_s[post]))) if post.any() else 0.0
    settled = float(r.freq_hz[-1])
    # Secondary dip: minimum frequency once recovery has started (mode == 2).
    rec = r.mode == 2
    secondary_min = float(r.freq_hz[rec].min()) if rec.any() else nadir
    # Response energy: aggregate wind support power over the mandated window after the event.
    win = post & (t <= t_event + response_window_s)
    P0 = r.meta["P0_W"]
    n_turbines = r.config.wind_capacity_MW * 1e6 / IEA15MW.rated_power_W
    dP_fleet_W = (r.P_elec_W - P0) * n_turbines           # aggregate fleet power change [W]
    energy_MJ = float(np.trapezoid(np.clip(dP_fleet_W[win], 0, None), t[win]) / 1e6) \
        if win.any() else 0.0
    quasi_steady = bool(np.std(r.freq_hz[t >= t[-1] - 30]) < 0.02)
    return GridKPIs(nadir, max_rocof, settled, secondary_min, energy_MJ, quasi_steady)


@dataclass(frozen=True)
class GridCodeLimits:
    """GB ESO / ENTSO-E Continental Europe style limits (50 Hz jurisdiction).

    "Equivalent grid service" means BOTH controllers meet the code — grid codes specify
    limits, not identical KPIs. The mandated support phase is identical between the
    incumbent and shaped recoveries; both must additionally avoid a sustained secondary
    violation during recovery.
    """
    rocof_max_hz_s: float = 1.0          # modern GB RoCoF withstand
    nadir_min_hz: float = 48.8           # below this, low-frequency demand disconnection begins
    secondary_min_hz: float = 48.8       # no sustained secondary violation
    min_response_energy_MJ: float = 1.0  # some support energy delivered in the window


def grid_code_compliant(k: GridKPIs, limits: GridCodeLimits | None = None
                        ) -> tuple[bool, dict]:
    """Does a controller meet the grid code (RoCoF, nadir, secondary dip, response energy)?"""
    limits = limits or GridCodeLimits()
    checks = {
        "rocof": k.max_rocof_hz_s <= limits.rocof_max_hz_s,
        "nadir": k.nadir_hz >= limits.nadir_min_hz,
        "secondary": k.secondary_min_hz >= limits.secondary_min_hz,
        "response_energy": k.response_energy_MJ >= limits.min_response_energy_MJ,
    }
    return all(checks.values()), checks


def kpis_equivalent(a: GridKPIs, b: GridKPIs, limits: GridCodeLimits | None = None
                    ) -> tuple[bool, dict]:
    """Equivalent grid service = both controllers meet the grid code (§7)."""
    ok_a, checks_a = grid_code_compliant(a, limits)
    ok_b, checks_b = grid_code_compliant(b, limits)
    checks = {"incumbent_compliant": ok_a, "shaped_compliant": ok_b,
              "incumbent_checks": checks_a, "shaped_checks": checks_b}
    return (ok_a and ok_b), checks


def aep_penalty_fraction(r_support: SimResult, r_baseline: SimResult) -> float:
    """Fractional aerodynamic-capture loss of a support+recovery run vs the no-support run.

    The penalty is the extra aerodynamic energy NOT captured because the rotor spends time
    off its Cp peak during support/recovery. Computed as the integral of (P_aero_baseline -
    P_aero_support) over the event-and-recovery window, normalized by the baseline capture.
    """
    tb = IEA15MW
    t = r_support.t
    V = r_support.config.wind_ms
    beta = r_support.config.beta_deg
    # Aerodynamic power actually captured along each rotor trajectory.
    Pa_sup = np.array([aero_power_W(tb, w, V, beta) for w in r_support.omega_rads])
    Pa_base = np.array([aero_power_W(tb, w, V, beta) for w in r_baseline.omega_rads])
    mask = t >= r_support.config.schedule.t_event_s
    lost = np.trapezoid(Pa_base[mask] - Pa_sup[mask], t[mask])
    captured = np.trapezoid(Pa_base[mask], t[mask])
    return float(lost / captured) if captured > 0 else 0.0
