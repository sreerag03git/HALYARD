"""Design-load-case (DLC) matrix + multi-seed UQ for annualized hang-off fatigue (§10, §5.8).

Industry fatigue assessment runs a matrix of environmental conditions (an IEC DLC 1.2-style
fatigue matrix) and aggregates the damage over the joint distribution. Here we run a reduced
but representative matrix:

  * wind speed bins, Weibull-weighted (the long-run occurrence of each wind speed);
  * a wind-correlated sea state (Hs, Tp) per bin (a documented representative correlation);
  * wave headings (wind-wave misalignment) with occurrence weights;
  * multiple wave seeds per condition -> a confidence band on the fatigue (UQ).

For each case we count on the COMBINED hang-off stress (wave + support) and the wave-only
baseline, annualize (continuous wave fatigue by hours-in-bin + control-added fatigue per
event), and report the mean life with a band from (a) the seed-to-seed scatter and (b) the
S-N curve's log-N standard deviation propagated to life.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from analysis.case import run_case
from models.metocean import SECONDS_PER_YEAR
from physics.cable import QuasiStaticFamily, build_quasistatic_family
from physics.fatigue import FatigueParams
from physics.simulation import SimConfig


# Representative wind-wave correlation and Weibull wind distribution.
def wind_weibull_weights(winds, A=10.0, k=2.0):
    """Occurrence weight of each wind-speed bin from a Weibull distribution (normalized)."""
    winds = np.asarray(winds, float)
    pdf = (k / A) * (winds / A) ** (k - 1) * np.exp(-(winds / A) ** k)
    return pdf / pdf.sum()


def sea_state_for_wind(V):
    """Representative wind-correlated sea state (Hs, Tp) [m, s]. Documented in ASSUMPTIONS."""
    Hs = max(0.5, 0.25 + 0.14 * V)
    Tp = 4.5 + 0.45 * V
    return Hs, Tp


@dataclass
class DLCResult:
    life_years_mean: float
    life_years_lo: float
    life_years_hi: float
    control_share_pct: float
    n_cases: int
    per_bin: list                    # list of dicts for display
    params: dict = field(default_factory=dict)


def run_dlc(base_cfg: SimConfig, events_per_year: float,
            winds=(8.0, 11.0, 14.0, 17.0), headings=(0.0, 45.0, 90.0),
            seeds=(1234, 5678), family: QuasiStaticFamily | None = None,
            fatigue_params: FatigueParams | None = None, daf: float = 1.2,
            hangoff_stick: bool = True, t_end: float = 250.0) -> DLCResult:
    """Run the reduced DLC matrix and aggregate the annualized hang-off life + band."""
    family = family or build_quasistatic_family()
    fp = fatigue_params or FatigueParams()
    wgt_wind = wind_weibull_weights(winds)
    head_wgt = np.ones(len(headings)) / len(headings)
    sn_logN_std = 0.20               # S-N scatter (log10 N) -> life band

    per_bin = []
    # accumulate annual damage per seed (for a seed-scatter band)
    annual_by_seed = {s: 0.0 for s in seeds}
    event_by_seed = {s: 0.0 for s in seeds}
    for iw, V in enumerate(winds):
        Hs, Tp = sea_state_for_wind(V)
        for ih, hd in enumerate(headings):
            w = wgt_wind[iw] * head_wgt[ih]
            d_wave_seeds, d_ws_seeds = [], []
            for s in seeds:
                cfg = replace(base_cfg, wind_ms=V, Hs_m=Hs, Tp_s=Tp, wave_seed=s,
                              wave_heading_deg=hd, t_end_s=t_end)
                wo = run_case(replace(cfg, enable_support=False), family, fp, daf, hangoff_stick)
                ws = run_case(replace(cfg, enable_support=True), family, fp, daf, hangoff_stick)
                Dwo = wo.fatigue["hang_off"].damage
                Dws = ws.fatigue["hang_off"].damage
                win = wo.fatigue["hang_off"].window_s
                d_wave_seeds.append(Dwo / win)          # damage rate
                d_ws_seeds.append(max(Dws - Dwo, 0.0))  # added per window
                annual_by_seed[s] += w * SECONDS_PER_YEAR * (Dwo / win)
                event_by_seed[s] += events_per_year * w * max(Dws - Dwo, 0.0)
            per_bin.append({"wind": V, "Hs": round(Hs, 2), "Tp": round(Tp, 1), "heading": hd,
                            "weight": round(w, 4),
                            "D_wave_rate": float(np.mean(d_wave_seeds)),
                            "D_added": float(np.mean(d_ws_seeds))})

    totals = np.array([annual_by_seed[s] + event_by_seed[s] for s in seeds])
    events = np.array([event_by_seed[s] for s in seeds])
    DFF = fp.DFF
    life = 1.0 / (totals * DFF)
    # Combine seed scatter with S-N log-N scatter (log-normal on life).
    life_mean = float(np.mean(life))
    seed_cv = float(np.std(life) / np.mean(life)) if np.mean(life) > 0 else 0.0
    ln_sigma = np.sqrt((np.log(10) * sn_logN_std) ** 2 + np.log(1 + seed_cv ** 2))
    lo = life_mean * np.exp(-1.645 * ln_sigma)
    hi = life_mean * np.exp(+1.645 * ln_sigma)
    control_share = 100 * float(np.mean(events)) / float(np.mean(totals)) if np.mean(totals) > 0 else 0.0
    return DLCResult(life_years_mean=life_mean, life_years_lo=lo, life_years_hi=hi,
                     control_share_pct=control_share, n_cases=len(winds) * len(headings) * len(seeds) * 2,
                     per_bin=per_bin,
                     params={"winds": list(winds), "headings": list(headings),
                             "seeds": list(seeds), "DFF": DFF, "sn_logN_std": sn_logN_std})
