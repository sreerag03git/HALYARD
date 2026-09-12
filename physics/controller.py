"""Frequency-support controller: support phase (§5.2) and recovery phase (§5.3).

The turbine is Type-4: the converter decouples the grid side across the DC link, so
support power can only come from the rotor's kinetic energy. The controller commands
an additive power modification dP_ctrl [pu of rating] on top of a baseline MPPT/rated
schedule; the generator torque follows, so the rotor slows (support) or is allowed to
re-accelerate (recovery).

Two phases, one design lever
----------------------------
* SUPPORT (grid-mandated, NOT shaped): synthetic inertia + droop with deadband, a
  magnitude clip, and a rate limit. Truncated if the KE reservoir would drive omega
  below omega_min (an undeliverable command must be truncated, not honoured).
* RECOVERY (the ONLY design lever, §5.3): the rotor is below setpoint and must
  re-accelerate. It does so by the controller holding electrical power *below*
  aerodynamic power (dP_ctrl < 0) so surplus aero torque spins the rotor up. The
  re-absorbed energy is fixed by the KE deficit; the free choice is the TIME-SHAPE of
  the re-absorption. A gentle (rate-limited, long-tau) return produces a small thrust
  transient (less hang-off fatigue) but keeps the rotor off its Cp peak longer (the AEP
  penalty). An abrupt return rings the platform. This is the trade HALYARD quantifies.

Incumbent baseline (cited, NOT a strawman)
------------------------------------------
The reference "incumbent" recovery is a fast first-order return of power once frequency
recovers, representative of published fast-frequency-response (FFR) schemes that
prioritise a quick restoration of output (e.g. the fast secondary-control return
described for synthetic-inertia wind schemes; see ASSUMPTIONS.md for the citation). It
is a legitimate, commonly-implemented recovery — HALYARD's shaped recovery is compared
against it on identical grid service.

Symbols / units
---------------
    H_wt   [s]    emulated (synthetic) inertia constant — a CONTROL gain, distinct from
                  the turbine's physical inertia constant
    R_droop[pu]   droop of the turbine's frequency response
    dP_max [pu]   converter short-term overload headroom for support
    rate   [pu/s] slew limit on dP_ctrl
    tau    [s]    first-order lag on dP_ctrl (converter/actuator + shaping)
    omega_min[rad/s] hard rotor-speed floor
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.iea15mw import Turbine


# --- Modes ------------------------------------------------------------------
MODE_IDLE = 0        # no event; dP_ctrl -> 0
MODE_SUPPORT = 1     # grid-mandated support
MODE_RECOVERY = 2    # rotor re-acceleration (shaped)


@dataclass(frozen=True)
class SupportParams:
    """Grid-mandated support-phase gains (NOT a design lever)."""
    H_wt_s: float = 6.0             # emulated inertia constant [s]
    R_droop: float = 0.05           # droop [pu]
    deadband_hz: float = 0.015      # frequency deadband [Hz] (~15 mHz)
    dP_max: float = 0.10            # magnitude limit [pu of rating]
    rate_pu_s: float = 0.50         # support slew limit [pu/s] (fast, grid-mandated)
    tau_s: float = 0.20             # converter/actuator lag [s]
    omega_margin_rads: float = 0.02  # stop extracting this far above omega_min


@dataclass(frozen=True)
class RecoveryParams:
    """Recovery-phase shape — THE design lever (§5.3).

    kind='incumbent' uses a fast return (cited baseline); kind='shaped' is HALYARD's
    tunable recovery. The 2-3 exposed knobs keep the sweep low-dimensional/transparent.
    """
    kind: str = "shaped"            # 'incumbent' | 'shaped'
    tau_rec_s: float = 12.0         # shaping time constant [s]  (large = gentle)
    rate_rec_pu_s: float = 0.02     # recovery slew limit [pu/s] (small = gentle)
    Kp_rec: float = 0.6             # under-production gain on normalized speed error [pu]
    dP_dip_max: float = 0.12        # max recovery under-production [pu]
    shape_exponent: float = 1.0     # 1 = plain; >1 eases the corner (half-cosine-like)

    @staticmethod
    def incumbent() -> "RecoveryParams":
        # Fast FFR-style return: short lag, high slew, aggressive gain.
        return RecoveryParams(kind="incumbent", tau_rec_s=2.0, rate_rec_pu_s=0.20,
                              Kp_rec=1.2, dP_dip_max=0.15, shape_exponent=1.0)


@dataclass(frozen=True)
class EventSchedule:
    """When support engages, and what triggers the support->recovery handover.

    A KE-limited turbine cannot hold droop support indefinitely: after a bounded support
    window it MUST begin recovering (re-accelerating) even while frequency is still
    depressed — which is exactly the mechanism of the secondary frequency dip (§5.3).
    Recovery therefore triggers on the FIRST of: (a) support window elapsed,
    (b) frequency recovered above df_clear, or (c) rotor near its speed floor.
    """
    t_event_s: float = 100.0        # disturbance time (after warm-up transient)
    df_engage_pu: float = -0.0006   # engage support below this df (~ -30 mHz)
    df_clear_pu: float = -0.0002    # frequency 'recovered' above this -> recovery (~ -10 mHz)
    min_support_s: float = 3.0      # minimum support duration (anti-chatter)
    support_window_s: float = 10.0  # max support duration before mandatory recovery
    omega_floor_margin_rads: float = 0.05  # force recovery if omega within this of floor
    recovery_done_frac: float = 0.990  # omega within this frac of setpoint -> idle


class FrequencyController:
    """Stateful controller: owns dP_ctrl dynamics and the support/recovery mode machine.

    The mode is updated once per fixed integration step (piecewise-constant across the
    RK4 sub-stages) to keep the ODE right-hand side smooth within a step.
    """

    def __init__(self, turbine: Turbine, support: SupportParams,
                 recovery: RecoveryParams, schedule: EventSchedule,
                 wind_ms: float, omega_setpoint_rads: float, P_baseline_pre_W: float):
        self.tb = turbine
        self.sp = support
        self.rp = recovery
        self.sch = schedule
        self.V = wind_ms
        self.omega_set = omega_setpoint_rads
        self.P_pre = P_baseline_pre_W
        # Mode-machine memory:
        self.mode = MODE_IDLE
        self._t_support_start = None
        # Cache the MPPT torque gain once (aero.optimal() is an argmax over the surface).
        lam_opt, _, cp_max = turbine.aero.optimal()
        self._K_opt = (0.5 * 1.225 * turbine.rotor_area_m2 * turbine.rotor_radius_m ** 3
                       * cp_max / lam_opt ** 3)                    # T = K_opt omega^2

    # -- baseline (normal) power schedule ------------------------------------
    def baseline_power_W(self, omega: float) -> float:
        """Type-4 MPPT-below-rated / rated-cap power schedule [W].

        Region-2 MPPT torque law T = K_opt omega^2, power capped at rated. K_opt from
        the calibrated Cp surface at (lambda_opt, Cp_max), cached in __init__.
        """
        P_mppt = self.tb.converter_efficiency * self._K_opt * omega ** 3
        return float(min(P_mppt, self.tb.rated_power_W))

    # -- mode machine (call once per step at step start) ---------------------
    def update_mode(self, t: float, df_pu: float, omega: float) -> None:
        s = self.sch
        if self.mode == MODE_IDLE:
            if t >= s.t_event_s and df_pu <= s.df_engage_pu:
                self.mode = MODE_SUPPORT
                self._t_support_start = t
        elif self.mode == MODE_SUPPORT:
            elapsed = t - (self._t_support_start or t)
            held = elapsed >= s.min_support_s
            near_floor = omega <= self.tb.omega_min_rads + s.omega_floor_margin_rads
            window_done = elapsed >= s.support_window_s
            if near_floor or (held and (df_pu >= s.df_clear_pu or window_done)):
                self.mode = MODE_RECOVERY
        elif self.mode == MODE_RECOVERY:
            if omega >= s.recovery_done_frac * self.omega_set and df_pu >= s.df_clear_pu:
                self.mode = MODE_IDLE

    # -- target power modification for the current mode ----------------------
    def dP_target_and_dynamics(self, df_pu: float, rocof_meas_hz_s: float,
                               omega: float) -> tuple[float, float, float]:
        """Return (dP_target [pu], tau [s], rate_limit [pu/s]) for the active mode."""
        f0 = 50.0
        if self.mode == MODE_SUPPORT:
            # Deadband on df (in Hz).
            df_hz = f0 * df_pu
            db = self.sp.deadband_hz
            df_hz_db = 0.0 if abs(df_hz) <= db else (df_hz - np.sign(df_hz) * db)
            dP_inertia = -2.0 * self.sp.H_wt_s * rocof_meas_hz_s / f0
            dP_droop = -(1.0 / self.sp.R_droop) * (df_hz_db / f0)
            dP = dP_inertia + dP_droop
            dP = float(np.clip(dP, -self.sp.dP_max, self.sp.dP_max))
            # Reservoir guard: never command support extraction if omega near the floor.
            if omega <= self.tb.omega_min_rads + self.sp.omega_margin_rads:
                dP = min(dP, 0.0)
            return dP, self.sp.tau_s, self.sp.rate_pu_s
        elif self.mode == MODE_RECOVERY:
            # Under-produce in proportion to the normalized speed deficit, shaped.
            err = (self.omega_set - omega) / self.omega_set
            err = max(err, 0.0)
            mag = self.rp.Kp_rec * (err ** self.rp.shape_exponent)
            dP = -min(mag, self.rp.dP_dip_max)
            return dP, self.rp.tau_rec_s, self.rp.rate_rec_pu_s
        else:  # IDLE
            return 0.0, self.sp.tau_s, self.sp.rate_pu_s

    def ddP_ctrl(self, dP_ctrl: float, df_pu: float, rocof_meas_hz_s: float,
                 omega: float) -> float:
        """d(dP_ctrl)/dt: first-order tracking of the target with a slew limit."""
        dP_target, tau, rate = self.dP_target_and_dynamics(df_pu, rocof_meas_hz_s, omega)
        d = (dP_target - dP_ctrl) / tau
        return float(np.clip(d, -rate, rate))

    def elec_power_W(self, omega: float, dP_ctrl: float) -> float:
        """Commanded electrical power [W] = baseline + dP_ctrl*rating (>= 0)."""
        P = self.baseline_power_W(omega) + dP_ctrl * self.tb.rated_power_W
        return max(P, 0.0)

    def delta_wind_pu(self, omega: float, dP_ctrl: float) -> float:
        """Change in turbine output vs its pre-event value [turbine pu] — feeds the grid."""
        return (self.elec_power_W(omega, dP_ctrl) - self.P_pre) / self.tb.rated_power_W
