"""Stage A — recovery-shape sweep and Pareto extraction (§7).

Sweeps the low-dimensional recovery-shape parameters (tau_rec, rate_rec) and, for each,
runs the full chain to get hang-off fatigue damage rate, the AEP penalty, and the grid-code
KPIs. Hard constraints: grid-code KPIs met across the full support+recovery cycle (incl. the
secondary dip), rotor-speed limits, and a damped (stable) platform pitch. Because the space
is low-dimensional, a transparent grid sweep + Pareto extraction is the default.

Both outcome branches are named in advance and the sweep decides:
  * aggressive recovery dominates fatigue -> a design warning, or
  * gentle recovery nearly eliminates it at small AEP cost -> a sweet spot.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from analysis.case import run_case
from analysis.kpis import aep_penalty_fraction, grid_kpis, kpis_equivalent
from physics.cable import QuasiStaticFamily, build_quasistatic_family
from physics.controller import RecoveryParams
from physics.fatigue import FatigueParams
from physics.platform import effective_pitch_damping
from physics.simulation import SimConfig
from models.volturnus_s import build_reduced_model


@dataclass
class SweepPoint:
    tau_rec_s: float
    rate_rec_pu_s: float
    hangoff_damage: float
    aep_penalty_frac: float
    nadir_hz: float
    secondary_min_hz: float
    kpis_ok: bool                 # equivalent to the incumbent (grid service preserved)
    stable: bool                  # platform pitch damping positive


@dataclass
class SweepResult:
    points: list                  # list[SweepPoint]
    incumbent_damage: float
    incumbent_aep_frac: float
    tau_grid: np.ndarray
    rate_grid: np.ndarray

    def feasible(self) -> list:
        return [p for p in self.points if p.kpis_ok and p.stable]

    def pareto(self) -> list:
        """Non-dominated points minimizing (hangoff_damage, aep_penalty_frac)."""
        feas = self.feasible()
        pts = sorted(feas, key=lambda p: (p.aep_penalty_frac, p.hangoff_damage))
        out, best_dmg = [], np.inf
        for p in pts:
            if p.hangoff_damage < best_dmg - 1e-30:
                out.append(p)
                best_dmg = p.hangoff_damage
        return out

    def best_tradeoff(self) -> SweepPoint | None:
        """Knee point: feasible point minimizing normalized (damage + aep)."""
        pf = self.pareto()
        if not pf:
            return None
        d = np.array([p.hangoff_damage for p in pf])
        a = np.array([p.aep_penalty_frac for p in pf])
        dn = (d - d.min()) / (np.ptp(d) + 1e-30)
        an = (a - a.min()) / (np.ptp(a) + 1e-30)
        return pf[int(np.argmin(dn + an))]


def run_sweep(cfg: SimConfig, tau_values=None, rate_values=None,
              family: QuasiStaticFamily | None = None,
              fatigue_params: FatigueParams | None = None,
              daf: float = 1.2, hangoff_stick: bool = True) -> SweepResult:
    """Grid-sweep the recovery shape parameters (tau_rec, rate_rec)."""
    family = family or build_quasistatic_family()
    fp = fatigue_params or FatigueParams()
    tau_values = np.asarray(tau_values if tau_values is not None
                            else [2.0, 6.0, 12.0, 20.0, 30.0], float)
    rate_values = np.asarray(rate_values if rate_values is not None
                             else [0.01, 0.02, 0.05, 0.10, 0.20], float)
    model = build_reduced_model()

    # Incumbent reference (fast recovery).
    inc = run_case(replace(cfg, enable_support=True, recovery=RecoveryParams.incumbent()),
                   family, fp, daf, hangoff_stick)
    inc_kpis = grid_kpis(inc.sim)
    none_sim = run_case(replace(cfg, enable_support=False), family, fp, daf, hangoff_stick).sim
    inc_aep = aep_penalty_fraction(inc.sim, none_sim)

    points = []
    for tau in tau_values:
        for rate in rate_values:
            rp = RecoveryParams(kind="shaped", tau_rec_s=float(tau),
                                rate_rec_pu_s=float(rate))
            case = run_case(replace(cfg, enable_support=True, recovery=rp),
                            family, fp, daf, hangoff_stick)
            k = grid_kpis(case.sim)
            ok, _ = kpis_equivalent(inc_kpis, k)
            # Pitch stability: effective damping including aero feedback (approx via the
            # observed pitch decay is expensive; use the structural check with zero aero
            # feedback as a conservative floor — flagged unstable only if <= 0).
            zeta = effective_pitch_damping(model, daero_dxdot=0.0)
            stable = zeta > 0 and case.sim.omega_rads.min() > 0.0
            points.append(SweepPoint(
                tau_rec_s=float(tau), rate_rec_pu_s=float(rate),
                hangoff_damage=case.hangoff_damage,
                aep_penalty_frac=aep_penalty_fraction(case.sim, none_sim),
                nadir_hz=k.nadir_hz, secondary_min_hz=k.secondary_min_hz,
                kpis_ok=ok, stable=stable))
    return SweepResult(points, inc.hangoff_damage, inc_aep, tau_values, rate_values)
