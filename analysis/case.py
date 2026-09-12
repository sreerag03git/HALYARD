"""Case runner: couple the simulation -> per-region cable stress -> fatigue (§5, §5.6-5.7).

One place that turns a ``SimConfig`` into hang-off (and other-region) stress time series
and fatigue results, so Phase-1, the recovery sweep, and the three-controller comparison
all use exactly the same chain. The quasi-static cable family is passed in (built and
cached once — it depends only on the frozen cable), keeping each case cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from physics.cable import (QuasiStaticFamily, build_quasistatic_family,
                           region_stress_timeseries, REGIONS)
from physics.fatigue import FatigueParams, FatigueResult, rainflow_damage
from physics.simulation import SimConfig, SimResult, run_simulation


@dataclass
class CaseResult:
    sim: SimResult
    stress_Pa: dict                  # region -> sigma(t) over the counting window
    curvature: dict                  # region -> kappa(t)
    tension_N: dict                  # region -> T(t)
    fatigue: dict                    # region -> FatigueResult
    t_count: np.ndarray              # time vector over the counting window
    daf: float
    hangoff_stick: bool

    @property
    def hangoff_damage(self) -> float:
        return self.fatigue["hang_off"].damage

    def dominant_region(self) -> str:
        return max(REGIONS, key=lambda r: self.fatigue[r].damage)


def run_case(cfg: SimConfig, family: QuasiStaticFamily | None = None,
             fatigue_params: FatigueParams | None = None, daf: float = 1.2,
             hangoff_stick: bool = True) -> CaseResult:
    """Run the full chain for one configuration and return per-region fatigue."""
    family = family or build_quasistatic_family()
    fatigue_params = fatigue_params or FatigueParams()
    sim = run_simulation(cfg)
    mask = sim.counting_mask
    t_count = sim.t[mask]
    window_s = float(t_count[-1] - t_count[0]) if len(t_count) > 1 else cfg.t_end_s
    surge = sim.surge_m
    pitch_rad = np.radians(sim.pitch_deg)

    stress, curv, tens, fat = {}, {}, {}, {}
    for reg in REGIONS:
        s, k, T = region_stress_timeseries(family, reg, surge, pitch_rad,
                                           daf=daf, hangoff_stick=hangoff_stick)
        stress[reg] = s[mask]
        curv[reg] = k[mask]
        tens[reg] = T[mask]
        fat[reg] = rainflow_damage(s[mask], window_s, fatigue_params)
    return CaseResult(sim=sim, stress_Pa=stress, curvature=curv, tension_N=tens,
                      fatigue=fat, t_count=t_count, daf=daf, hangoff_stick=hangoff_stick)
