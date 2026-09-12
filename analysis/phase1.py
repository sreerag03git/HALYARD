"""Phase-1 spectral-overlap gate — the honesty mechanism (§6, implemented FIRST in spirit).

Before any sweep or comparison, decide whether the control-driven cable-fatigue effect can
even exist under the current inputs. Thresholds are PRE-REGISTERED in a config file and
loaded (committed before results are seen). If the gate FAILS, the app must state plainly
"under these conditions the effect is negligible", show the evidence, and NOT unlock a saving.

Gate logic (all evaluated at the hang-off, the fatigue-critical region):
  1. added-damage fraction = (D_wave+support - D_wave-only) / D_wave+support  >  X
     (does frequency support add a meaningful share of Miner damage vs the wave baseline?)
  2. recovery share = slow-drift stress excursion during the recovery phase /
     (support + recovery excursion)                                            >  Y
     (is the shapeable recovery phase actually where the slow-drift damage sits?)
  3. fatigue-relevant amplitude: peak slow-drift stress excursion during the event > min
     (guards against passing on spectral coincidence alone when amplitudes are tiny)

Evidence returned: PSDs of hang-off stress (wave-only vs wave+support) and the platform
modal bands, for the overlap plot.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from analysis.case import CaseResult, run_case
from physics.cable import QuasiStaticFamily, build_quasistatic_family
from physics.constants import PA_TO_MPA
from physics.fatigue import FatigueParams
from physics.simulation import SimConfig
from models.volturnus_s import build_reduced_model


@dataclass(frozen=True)
class Phase1Thresholds:
    """Pre-registered gate thresholds (loaded from a committed config)."""
    added_damage_frac_min: float = 0.05     # X: control must add > 5% of hang-off damage
    recovery_share_min: float = 0.20        # Y: recovery must hold > 20% of slow-drift excursion
    min_slow_stress_amp_MPa: float = 0.5    # fatigue-relevant amplitude floor
    slow_cutoff_hz: float = 0.03            # low-pass cutoff separating drift from waves


@dataclass
class Phase1Result:
    passed: bool
    added_damage_frac: float
    recovery_share: float
    slow_stress_amp_MPa: float
    reasons: dict                            # which sub-criteria passed
    thresholds: Phase1Thresholds
    # evidence for plotting
    psd_freq: np.ndarray = field(default_factory=lambda: np.array([]))
    psd_wave_only: np.ndarray = field(default_factory=lambda: np.array([]))
    psd_with_support: np.ndarray = field(default_factory=lambda: np.array([]))
    modal_bands_hz: dict = field(default_factory=dict)
    D_wave_only: float = 0.0
    D_with_support: float = 0.0
    case_support: CaseResult | None = None
    case_wave_only: CaseResult | None = None

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "NEGLIGIBLE"


def _lowpass(x: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    fs = 1.0 / dt
    wn = min(cutoff_hz / (0.5 * fs), 0.99)
    b, a = butter(3, wn, btype="low")
    return filtfilt(b, a, x)


def _welch_psd(x: np.ndarray, dt: float):
    from scipy.signal import welch
    fs = 1.0 / dt
    nperseg = min(len(x), 4096)
    f, p = welch(x - np.mean(x), fs=fs, nperseg=nperseg)
    return f, p


def run_phase1(cfg: SimConfig, thresholds: Phase1Thresholds | None = None,
               family: QuasiStaticFamily | None = None,
               fatigue_params: FatigueParams | None = None,
               daf: float = 1.2, hangoff_stick: bool = True) -> Phase1Result:
    """Run the Phase-1 gate for a configuration. Returns PASS/NEGLIGIBLE + evidence."""
    thresholds = thresholds or Phase1Thresholds()
    family = family or build_quasistatic_family()
    fp = fatigue_params or FatigueParams()

    cfg_ws = replace(cfg, enable_support=True)
    cfg_wo = replace(cfg, enable_support=False)
    case_ws = run_case(cfg_ws, family, fp, daf, hangoff_stick)
    case_wo = run_case(cfg_wo, family, fp, daf, hangoff_stick)

    D_ws = case_ws.fatigue["hang_off"].damage
    D_wo = case_wo.fatigue["hang_off"].damage
    added_frac = (D_ws - D_wo) / D_ws if D_ws > 0 else 0.0

    # Slow-drift stress at the hang-off (low-pass below the wave band).
    dt = cfg.dt_s
    sig_ws = case_ws.stress_Pa["hang_off"]
    slow = _lowpass(sig_ws, dt, thresholds.slow_cutoff_hz) * PA_TO_MPA

    # Recovery share: slow-drift excursion during recovery vs support windows.
    mode = case_ws.sim.mode[case_ws.sim.counting_mask]
    supp = slow[mode == 1]
    rec = slow[mode == 2]
    exc_supp = float(np.ptp(supp)) if supp.size > 2 else 0.0
    exc_rec = float(np.ptp(rec)) if rec.size > 2 else 0.0
    recovery_share = exc_rec / (exc_supp + exc_rec) if (exc_supp + exc_rec) > 0 else 0.0

    # Fatigue-relevant amplitude: peak slow-drift excursion during the event.
    event_mask = case_ws.t_count >= cfg.schedule.t_event_s
    slow_amp = float(np.ptp(slow[event_mask])) if event_mask.any() else 0.0

    # PSD evidence.
    f_wo, p_wo = _welch_psd(case_wo.stress_Pa["hang_off"] * PA_TO_MPA, dt)
    f_ws, p_ws = _welch_psd(sig_ws * PA_TO_MPA, dt)
    model = build_reduced_model()
    per = np.sort(model.natural_periods_s())
    modal = {"surge": 1.0 / per[-1], "pitch": 1.0 / per[0]}

    reasons = {
        "added_damage": added_frac > thresholds.added_damage_frac_min,
        "recovery_share": recovery_share > thresholds.recovery_share_min,
        "slow_amplitude": slow_amp > thresholds.min_slow_stress_amp_MPa,
    }
    passed = all(reasons.values())
    return Phase1Result(
        passed=passed, added_damage_frac=added_frac, recovery_share=recovery_share,
        slow_stress_amp_MPa=slow_amp, reasons=reasons, thresholds=thresholds,
        psd_freq=f_ws, psd_wave_only=np.interp(f_ws, f_wo, p_wo), psd_with_support=p_ws,
        modal_bands_hz=modal, D_wave_only=D_wo, D_with_support=D_ws,
        case_support=case_ws, case_wave_only=case_wo)
