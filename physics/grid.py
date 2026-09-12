"""Reduced-order grid frequency — System Frequency Response (SFR) / swing equation (§5.1).

Physics
-------
Treat the whole synchronous grid as one equivalent rotating machine whose speed is the
system frequency. A power imbalance accelerates/decelerates that inertia, so frequency
swings. Low system inertia H_sys -> steeper RoCoF and deeper nadir: precisely the regime
where fast frequency support is demanded of wind, and where HALYARD's effect is largest.

State (per-unit on the system base, except RoCoF which we carry in Hz/s):
    df   : frequency deviation Delta f  [pu]      (Delta f_Hz = f0 * df)
    pgov : aggregate primary (governor) response  [pu]  — arrests the nadir
    rocof: *measured* RoCoF  [Hz/s]               — low-pass of the true RoCoF (PLL/measurement lag)

Governing equations
-------------------
    2 H_sys d(df)/dt = pgov + p_wind - p_load - D df           (swing)
    T_gov  d(pgov)/dt = -pgov - (1/R_sys) df                    (primary response, reserve-limited)
    T_roc  d(rocof)/dt = rocof_true - rocof                     (measurement filter)
where rocof_true = f0 * d(df)/dt [Hz/s], p_wind is the aggregate wind-fleet support
contribution (participation * per-turbine Delta P), and p_load is the triggering imbalance.

The measured-RoCoF state deliberately breaks the algebraic loop that synthetic inertia
(which reacts to RoCoF, §5.2) would otherwise form with the swing equation — and it is
physically correct: real controllers act on a filtered RoCoF, never the instantaneous one.

Symbols / units
---------------
    H_sys [s]   aggregate inertia constant (strong 4-6; low-inertia 1.5-3; islanded <1.5)
    D     [pu/pu] load damping (~1-2, i.e. ~2-4 %/Hz)
    R_sys [pu]  aggregate governor droop of the *rest* of the system (~0.05)
    T_gov [s]   effective primary-response lag (governor + prime-mover)
    reserve[pu] spinning reserve available to primary response
    T_roc [s]   RoCoF-measurement filter time constant
    S_base_MW   system base power (sets what "pu" means)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.constants import F0_HZ

# State layout within the grid sub-vector.
IDX_DF = 0
IDX_PGOV = 1
IDX_ROCOF = 2
N_GRID_STATES = 3


@dataclass(frozen=True)
class GridModel:
    """Aggregated single-bus SFR model of a (low-inertia) power system."""

    H_sys_s: float = 2.5          # aggregate inertia constant [s] (low-inertia default)
    D_load: float = 1.0           # load damping [pu/pu]
    R_sys: float = 0.05           # rest-of-system governor droop [pu]
    T_gov_s: float = 8.0          # aggregate primary-response lag [s]
    reserve_pu: float = 0.12      # spinning reserve for primary response [pu]
    T_rocof_s: float = 0.5        # RoCoF measurement-filter time constant [s]
    S_base_MW: float = 3000.0     # system base [MW] (low-inertia island/region)
    f0_Hz: float = F0_HZ

    def rhs(self, grid_state: np.ndarray, p_wind_pu: float, p_load_pu: float
            ) -> tuple[np.ndarray, float]:
        """Return (d/dt of grid states, true RoCoF in Hz/s).

        p_wind_pu : aggregate wind-support power on the *system* base (already scaled
                    by participation), added to generation.
        p_load_pu : triggering imbalance on the system base (a generation loss is +ve
                    p_load, i.e. a deficit that pulls frequency down).
        """
        df = grid_state[IDX_DF]
        pgov = grid_state[IDX_PGOV]
        rocof_meas = grid_state[IDX_ROCOF]

        # Swing equation -> d(df)/dt.
        ddf = (pgov + p_wind_pu - p_load_pu - self.D_load * df) / (2.0 * self.H_sys_s)
        rocof_true = self.f0_Hz * ddf   # Hz/s

        # Primary governor response, reserve-limited (soft clip via target then clamp).
        pgov_target = -(1.0 / self.R_sys) * df
        dpgov = (pgov_target - pgov) / self.T_gov_s
        # Reserve limit: hold pgov within +/- reserve.
        if (pgov >= self.reserve_pu and dpgov > 0) or (pgov <= -self.reserve_pu and dpgov < 0):
            dpgov = 0.0

        # RoCoF measurement filter.
        drocof = (rocof_true - rocof_meas) / self.T_rocof_s

        d = np.empty(N_GRID_STATES)
        d[IDX_DF] = ddf
        d[IDX_PGOV] = dpgov
        d[IDX_ROCOF] = drocof
        return d, rocof_true

    # -- convenience accessors -------------------------------------------------
    def freq_hz(self, df_pu: float) -> float:
        return self.f0_Hz * (1.0 + df_pu)

    def initial_state(self) -> np.ndarray:
        return np.zeros(N_GRID_STATES)


def load_participation(wind_capacity_MW: float, grid: GridModel) -> float:
    """Fraction of the system that is identical IEA-15MW turbines providing support.

    All supporting turbines see the same Delta f, so the aggregate system-base
    contribution is participation * (per-turbine Delta P in turbine pu). One
    representative turbine's motion is analysed for fatigue; the fleet drives the grid.
    """
    return float(wind_capacity_MW) / float(grid.S_base_MW)
