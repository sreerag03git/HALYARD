"""Irregular wave field — JONSWAP spectrum and a seeded random realization (§5.6 build).

We build a JONSWAP sea from (Hs, Tp) and synthesize an irregular surface elevation and
its linear (Airy) kinematics as a sum of N components with seeded random phases. This is
the wave excitation for the platform (Morison/Froude-Krylov forces are formed in
``platform.py``). The realization is deterministic given the seed (reproducibility, §11).

JONSWAP one-sided spectrum (Hasselmann et al.):
    S(f) = (5/16) Hs^2 fp^4 f^-5 exp[-1.25 (fp/f)^4] gamma^r (1 - 0.287 ln gamma)
    r    = exp[-(f - fp)^2 / (2 sigma^2 fp^2)],  sigma = 0.07 (f<=fp) else 0.09
so that the zeroth moment m0 = integral S df = Hs^2/16 and Hs = 4 sqrt(m0).

Component amplitudes a_i = sqrt(2 S(f_i) df_i); elevation eta(t,x) = sum a_i cos(w_i t - k_i x + phi_i).
Stratified random frequency sampling (one random draw per bin) removes the signal's
periodic repetition over a finite run. Deep-water dispersion (w^2 = g k) is used: at the
200 m reference site, wave periods of interest (~5-16 s) are deep-water (k d >> 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.constants import G


@dataclass
class WaveField:
    Hs_m: float
    Tp_s: float
    gamma: float = 3.3
    seed: int = 1234
    n_components: int = 200
    f_min_hz: float = 0.02
    f_max_hz: float = 0.40

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        edges = np.linspace(self.f_min_hz, self.f_max_hz, self.n_components + 1)
        # Stratified sampling: one random frequency per bin (breaks periodicity).
        u = rng.random(self.n_components)
        self.f = edges[:-1] + u * (edges[1:] - edges[:-1])
        self.df = edges[1:] - edges[:-1]
        self.w = 2.0 * np.pi * self.f
        self.k = self.w ** 2 / G                      # deep-water dispersion
        S = self._jonswap(self.f)
        self.amp = np.sqrt(2.0 * S * self.df)
        self.phase = rng.uniform(0.0, 2.0 * np.pi, self.n_components)
        # Verify the realization reproduces Hs (sanity, used in tests/validation).
        self.m0 = float(np.sum(S * self.df))
        self.Hs_realized = 4.0 * np.sqrt(self.m0)

    def _jonswap(self, f: np.ndarray) -> np.ndarray:
        fp = 1.0 / self.Tp_s
        f = np.asarray(f, float)
        with np.errstate(divide="ignore", over="ignore"):
            sigma = np.where(f <= fp, 0.07, 0.09)
            r = np.exp(-((f - fp) ** 2) / (2.0 * sigma ** 2 * fp ** 2))
            base = (5.0 / 16.0) * self.Hs_m ** 2 * fp ** 4 * f ** -5.0 \
                * np.exp(-1.25 * (fp / f) ** 4)
            norm = 1.0 - 0.287 * np.log(self.gamma)
            S = base * self.gamma ** r * norm
        S[~np.isfinite(S)] = 0.0
        return S

    # --- kinematics (linear Airy, deep water) at horizontal position x, depth z<=0 ---
    def elevation(self, t: float, x: float = 0.0) -> float:
        return float(np.sum(self.amp * np.cos(self.w * t - self.k * x + self.phase)))

    def horizontal_acceleration(self, t: float, x: float, z: float) -> float:
        """du/dt at (x, z<=0) [m/s^2]. a_x = sum a w^2 e^{kz} sin(w t - k x + phi)."""
        decay = np.exp(self.k * z)   # z<=0
        return float(np.sum(self.amp * self.w ** 2 * decay
                            * np.sin(self.w * t - self.k * x + self.phase)))

    def horizontal_velocity(self, t: float, x: float, z: float) -> float:
        decay = np.exp(self.k * z)
        return float(np.sum(self.amp * self.w * decay
                            * np.cos(self.w * t - self.k * x + self.phase)))

    def spectrum(self, f: np.ndarray) -> np.ndarray:
        return self._jonswap(f)
