"""Representative metocean sea-state scatter for fatigue annualization (§4).

The full wave scatter (a large Hs x Tp occurrence table) is reduced to a small set of
fatigue-relevant bins. This is a **[representative]** reduction disclosed here and in
ASSUMPTIONS.md — a typical North-Sea-like distribution weighted toward the frequent,
lower-Hs states where the control-driven cable-fatigue effect is most relevant (calm seas
do not swamp the slow control drift). Occurrences sum to 1.0.

Each bin gives (Hs, Tp, occurrence). JONSWAP spectra per bin come from ``physics.waves``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeaState:
    Hs_m: float
    Tp_s: float
    occurrence: float           # fraction of the year


# Reduced 7-bin scatter (representative). Weighted toward frequent calm/moderate seas.
SCATTER = (
    SeaState(0.75, 6.5, 0.22),
    SeaState(1.25, 7.5, 0.24),
    SeaState(1.75, 8.5, 0.20),
    SeaState(2.50, 9.5, 0.16),
    SeaState(3.50, 10.5, 0.10),
    SeaState(4.50, 11.5, 0.06),
    SeaState(6.00, 12.5, 0.02),
)

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


def check_occurrence_sums_to_one() -> float:
    return sum(s.occurrence for s in SCATTER)
