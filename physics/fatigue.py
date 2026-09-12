"""Fatigue — rainflow -> mean-stress correction -> S-N -> Miner, on the COMBINED signal (§5.7).

The novelty lives in the interaction: the slow control-driven drift does not add its own
separate cycles so much as it shifts the MEAN of the fast wave cycles. A wave cycle sitting
on a higher mean does more damage (an R-ratio effect). This is why counting MUST be on the
single combined stress signal and never as "wave-only damage + control-only damage" — that
sum is physically wrong and would be the study's fatal error.

Pipeline (per critical region), all on sigma_combined(t):
  1. Rainflow count -> {stress range S_i, mean sigma_m,i, count n_i}   (ASTM E1049)
  2. Mean-stress correction to an equivalent fully-reversed range:
       Goodman: S_eq = S / (1 - sigma_m/sigma_u)          (primary)
       Gerber : S_eq = S / (1 - (sigma_m/sigma_u)^2)      (sensitivity toggle)
  3. S-N (bilinear): log10 N = log_a - m log10 S_eq       (DNV-RP-C203-style)
  4. Miner: D = sum n_i / N_i
  5. Damage rate -> annualize (occurrence / repeats) -> life = 1 / annual_damage
  6. Design Fatigue Factor (DFF) reduces the allowable life (§5.8).

Stresses are handled in MPa inside the S-N law (curve constants are MPa-based).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rainflow

from physics.constants import PA_TO_MPA


@dataclass(frozen=True)
class SNCurve:
    """Bilinear S-N curve, MPa-based: log10 N = log_a(seg) - m(seg) * log10(S_eq).

    Default is a DNV-RP-C203-style two-slope steel curve (representative of the armour-wire
    steel in seawater with cathodic protection). Source/label in ASSUMPTIONS.md.
    """
    m1: float = 3.0
    log_a1: float = 11.764        # N < N_knee (MPa units)
    m2: float = 5.0
    log_a2: float = 15.606        # N > N_knee
    N_knee: float = 1.0e6
    sigma_u_MPa: float = 1400.0   # armour-wire UTS (Goodman/Gerber mean-stress correction)
    endurance_limit_MPa: float = 0.0   # 0 => no cut-off (count all ranges)

    def cycles_to_failure(self, S_eq_MPa: np.ndarray) -> np.ndarray:
        """N(S_eq) from the bilinear curve. Below the endurance limit -> infinite N."""
        S = np.asarray(S_eq_MPa, float)
        S_knee = 10.0 ** ((self.log_a1 - np.log10(self.N_knee)) / self.m1)
        with np.errstate(divide="ignore"):
            logN = np.where(S >= S_knee,
                            self.log_a1 - self.m1 * np.log10(np.maximum(S, 1e-9)),
                            self.log_a2 - self.m2 * np.log10(np.maximum(S, 1e-9)))
        N = 10.0 ** logN
        if self.endurance_limit_MPa > 0:
            N = np.where(S < self.endurance_limit_MPa, np.inf, N)
        return N


@dataclass
class FatigueParams:
    sn: SNCurve = field(default_factory=SNCurve)
    mean_stress: str = "goodman"   # 'goodman' | 'gerber' | 'none'
    DFF: float = 3.0               # design fatigue factor (critical/inaccessible: 3-10)


@dataclass
class FatigueResult:
    ranges_MPa: np.ndarray
    means_MPa: np.ndarray
    counts: np.ndarray
    S_eq_MPa: np.ndarray
    N_i: np.ndarray
    damage: float                  # Miner sum over the analysed window
    window_s: float
    n_cycles: float
    params: FatigueParams

    def annual_damage(self, repeats_per_year: float, occurrence: float = 1.0) -> float:
        """Annual damage = window damage * (events/year or continuous scaling) * occurrence."""
        return self.damage * repeats_per_year * occurrence

    def life_years(self, repeats_per_year: float, occurrence: float = 1.0) -> float:
        d = self.annual_damage(repeats_per_year, occurrence)
        if d <= 0:
            return np.inf
        return 1.0 / (d * self.params.DFF)


def mean_stress_correct(S: np.ndarray, mean: np.ndarray, sigma_u_MPa: float,
                        method: str) -> np.ndarray:
    """Convert stress ranges to equivalent fully-reversed ranges (mean-stress correction)."""
    S = np.asarray(S, float)
    m = np.asarray(mean, float)
    method = method.lower()
    if method == "none":
        return S
    # Only tensile mean increases damage; compressive mean is (conservatively) not benefit.
    r = np.clip(m / sigma_u_MPa, 0.0, 0.95)   # cap to avoid singularity/negatives
    if method == "goodman":
        return S / (1.0 - r)
    if method == "gerber":
        return S / (1.0 - r ** 2)
    raise ValueError(f"unknown mean-stress method: {method}")


def rainflow_damage(sigma_Pa: np.ndarray, window_s: float,
                    params: FatigueParams | None = None) -> FatigueResult:
    """Full rainflow -> mean-stress -> S-N -> Miner on a combined stress signal [Pa]."""
    params = params or FatigueParams()
    sigma_MPa = np.asarray(sigma_Pa, float) * PA_TO_MPA
    cycles = list(rainflow.extract_cycles(sigma_MPa))
    if not cycles:
        return FatigueResult(np.array([]), np.array([]), np.array([]), np.array([]),
                             np.array([]), 0.0, window_s, 0.0, params)
    arr = np.array(cycles)                      # columns: range, mean, count, i_start, i_end
    ranges, means, counts = arr[:, 0], arr[:, 1], arr[:, 2]
    S_eq = mean_stress_correct(ranges, means, params.sn.sigma_u_MPa, params.mean_stress)
    N_i = params.sn.cycles_to_failure(S_eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dmg = np.where(np.isfinite(N_i) & (N_i > 0), counts / N_i, 0.0)
    return FatigueResult(ranges_MPa=ranges, means_MPa=means, counts=counts,
                         S_eq_MPa=S_eq, N_i=N_i, damage=float(np.sum(dmg)),
                         window_s=window_s, n_cycles=float(np.sum(counts)), params=params)


def rainflow_histogram(result: FatigueResult, n_bins: int = 30
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Range histogram (bin centres [MPa], summed counts) for plotting."""
    if len(result.ranges_MPa) == 0:
        return np.array([]), np.array([])
    hist, edges = np.histogram(result.ranges_MPa, bins=n_bins, weights=result.counts)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, hist
