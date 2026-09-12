"""Stage B — the headline three-controller comparison (§7).

Runs the identical pipeline for three controllers on the same wave seed:
  (1) no support            — the wave-only fatigue baseline
  (2) incumbent recovery    — a cited fast FFR-style return (NOT a strawman)
  (3) HALYARD shaped recovery — the fatigue-aware recovery

CREDIBILITY GATE: (2) and (3) must first be shown *electrically equivalent* — grid-code
KPIs match within tolerance, secondary dip included. Only then is the cable-fatigue
comparison meaningful. If they are not equivalent, the comparison is void and the app says
so. The single defensible headline is: same grid service, measurably different hang-off
fatigue, at a quantified AEP trade — nothing stronger.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from analysis.case import CaseResult, run_case
from analysis.kpis import GridKPIs, aep_penalty_fraction, grid_kpis, kpis_equivalent
from physics.cable import QuasiStaticFamily, build_quasistatic_family
from physics.controller import RecoveryParams
from physics.fatigue import FatigueParams
from physics.simulation import SimConfig


@dataclass
class ControllerRun:
    label: str
    case: CaseResult
    kpis: GridKPIs
    aep_penalty_frac: float


@dataclass
class ComparisonResult:
    no_support: ControllerRun
    incumbent: ControllerRun
    halyard: ControllerRun
    equivalent: bool
    equivalence_checks: dict
    hangoff_damage_reduction_frac: float     # (incumbent - halyard) / incumbent, at hang-off

    def summary(self) -> dict:
        return {
            "equivalent": self.equivalent,
            "equivalence_checks": self.equivalence_checks,
            "D_hangoff_no_support": self.no_support.case.hangoff_damage,
            "D_hangoff_incumbent": self.incumbent.case.hangoff_damage,
            "D_hangoff_halyard": self.halyard.case.hangoff_damage,
            "hangoff_damage_reduction_%": 100 * self.hangoff_damage_reduction_frac,
            "aep_penalty_incumbent_%": 100 * self.incumbent.aep_penalty_frac,
            "aep_penalty_halyard_%": 100 * self.halyard.aep_penalty_frac,
        }


def run_comparison(cfg: SimConfig, halyard_recovery: RecoveryParams | None = None,
                   family: QuasiStaticFamily | None = None,
                   fatigue_params: FatigueParams | None = None,
                   daf: float = 1.2, hangoff_stick: bool = True) -> ComparisonResult:
    """Run the three-controller comparison with the electrical-equivalence gate."""
    family = family or build_quasistatic_family()
    fp = fatigue_params or FatigueParams()
    halyard_recovery = halyard_recovery or RecoveryParams()   # default shaped

    cfg_none = replace(cfg, enable_support=False)
    cfg_inc = replace(cfg, enable_support=True, recovery=RecoveryParams.incumbent())
    cfg_hal = replace(cfg, enable_support=True, recovery=halyard_recovery)

    none_case = run_case(cfg_none, family, fp, daf, hangoff_stick)
    none_run = ControllerRun("no_support", none_case, grid_kpis(none_case.sim), 0.0)
    inc_case = run_case(cfg_inc, family, fp, daf, hangoff_stick)
    hal_case = run_case(cfg_hal, family, fp, daf, hangoff_stick)
    inc_run = ControllerRun("incumbent", inc_case, grid_kpis(inc_case.sim),
                            aep_penalty_fraction(inc_case.sim, none_case.sim))
    hal_run = ControllerRun("halyard", hal_case, grid_kpis(hal_case.sim),
                            aep_penalty_fraction(hal_case.sim, none_case.sim))

    equivalent, checks = kpis_equivalent(inc_run.kpis, hal_run.kpis)
    D_inc = inc_run.case.hangoff_damage
    D_hal = hal_run.case.hangoff_damage
    reduction = (D_inc - D_hal) / D_inc if D_inc > 0 else 0.0
    return ComparisonResult(none_run, inc_run, hal_run, equivalent, checks, reduction)
