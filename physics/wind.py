"""Turbulent inflow — IEC 61400-1 Kaimal longitudinal wind spectrum (§5.8 aero extension).

Generates a seeded hub-height longitudinal wind time series U(t) = U_mean + u'(t) whose
fluctuations follow the Kaimal spectrum (a TurbSim-equivalent point spectrum):

    S_u(f) = 4 sigma_u^2 (L_u/U) / (1 + 6 f L_u/U)^(5/3)

with sigma_u = TI * U_mean and length scale L_u = 8.1 * Lambda_1, Lambda_1 = 0.7*min(60, z_hub).
Realized by summing frequency components with seeded random phases (deterministic).

Turbulent inflow is OFF by default: steady wind isolates the control-driven effect from
turbulence noise (which would need multi-seed averaging to see the control signal). It is a
toggle so the incremental effect of turbulence is inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TurbulentWind:
    U_mean: float
    TI: float = 0.10                 # turbulence intensity [-]
    z_hub: float = 150.0
    seed: int = 2024
    f_min_hz: float = 0.002
    f_max_hz: float = 2.0
    n_components: int = 400

    def series(self, t: np.ndarray) -> np.ndarray:
        """Return U(t) [m/s] on the given time grid (>= 0)."""
        rng = np.random.default_rng(self.seed)
        sigma = self.TI * self.U_mean
        Lu = 8.1 * 0.7 * min(60.0, self.z_hub)
        edges = np.linspace(self.f_min_hz, self.f_max_hz, self.n_components + 1)
        f = 0.5 * (edges[:-1] + edges[1:])
        df = edges[1:] - edges[:-1]
        x = f * Lu / self.U_mean
        S = 4.0 * sigma ** 2 * (Lu / self.U_mean) / (1.0 + 6.0 * x) ** (5.0 / 3.0)
        amp = np.sqrt(2.0 * S * df)
        ph = rng.uniform(0, 2 * np.pi, self.n_components)
        w = 2 * np.pi * f
        u = (amp[None, :] * np.cos(np.outer(t, w) + ph[None, :])).sum(axis=1)
        U = self.U_mean + u
        return np.maximum(U, 0.5)

    def realized_TI(self, t: np.ndarray) -> float:
        U = self.series(t)
        return float(np.std(U) / np.mean(U))
