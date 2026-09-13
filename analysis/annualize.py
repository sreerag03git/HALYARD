"""Annualized hang-off fatigue life across the sea-state scatter (§5.7 step 5).

A finite simulation is extrapolated to a lifetime by (a) scaling the continuous wave-driven
damage of each sea-state bin by the hours spent in that bin, and (b) adding the extra damage
each frequency-support event causes, at the rate of events per year, distributed across the
bins by occurrence. Counting is always on the COMBINED signal per bin (never summed).

  annual_damage = Σ_bins occ · [ (year/window)·D_wave  +  events/yr·(D_support − D_wave) ]
  life_years    = 1 / (annual_damage · DFF)

The event term uses (D_support − D_wave) so only the *added* damage from control is charged
per event; the continuous term is the ordinary wave fatigue. The scaling is disclosed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from analysis.case import run_case
from models.metocean import SCATTER, SECONDS_PER_YEAR, SeaState
from physics.cable import QuasiStaticFamily, build_quasistatic_family
from physics.fatigue import FatigueParams
from physics.simulation import SimConfig


@dataclass
class BinResult:
    sea: SeaState
    D_wave: float
    D_support: float
    added: float


@dataclass
class AnnualResult:
    bins: list
    continuous_annual_damage: float
    event_annual_damage: float
    total_annual_damage: float
    life_years: float
    life_years_wave_only: float
    events_per_year: float
    DFF: float


def annualized_life(cfg: SimConfig, events_per_year: float,
                    family: QuasiStaticFamily | None = None,
                    fatigue_params: FatigueParams | None = None, daf: float = 1.2,
                    hangoff_stick: bool = True, scatter=SCATTER) -> AnnualResult:
    """Run the pipeline for each sea-state bin and extrapolate to a lifetime."""
    family = family or build_quasistatic_family()
    fp = fatigue_params or FatigueParams()
    bins = []
    cont_annual = 0.0
    event_annual = 0.0
    cont_annual_wave = 0.0
    for s in scatter:
        c_ws = replace(cfg, Hs_m=s.Hs_m, Tp_s=s.Tp_s, enable_support=True)
        c_wo = replace(cfg, Hs_m=s.Hs_m, Tp_s=s.Tp_s, enable_support=False)
        case_ws = run_case(c_ws, family, fp, daf, hangoff_stick)
        case_wo = run_case(c_wo, family, fp, daf, hangoff_stick)
        window = case_ws.fatigue["hang_off"].window_s
        D_ws = case_ws.fatigue["hang_off"].damage
        D_wo = case_wo.fatigue["hang_off"].damage
        added = max(D_ws - D_wo, 0.0)
        bins.append(BinResult(s, D_wo, D_ws, added))
        # Continuous wave damage scaled to hours in this bin.
        cont_annual += s.occurrence * (SECONDS_PER_YEAR / window) * D_wo
        cont_annual_wave += s.occurrence * (SECONDS_PER_YEAR / window) * D_wo
        # Event-added damage: events distributed across bins by occurrence.
        event_annual += events_per_year * s.occurrence * added

    total = cont_annual + event_annual
    DFF = fp.DFF
    life = 1.0 / (total * DFF) if total > 0 else float("inf")
    life_wave = 1.0 / (cont_annual_wave * DFF) if cont_annual_wave > 0 else float("inf")
    return AnnualResult(bins=bins, continuous_annual_damage=cont_annual,
                        event_annual_damage=event_annual, total_annual_damage=total,
                        life_years=life, life_years_wave_only=life_wave,
                        events_per_year=events_per_year, DFF=DFF)
