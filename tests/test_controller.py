"""Frequency controller: support law, reservoir guard, recovery direction."""
import numpy as np

from models.iea15mw import IEA15MW
from physics.controller import (EventSchedule, FrequencyController, RecoveryParams,
                                SupportParams, MODE_SUPPORT, MODE_RECOVERY)


def _ctl():
    tb = IEA15MW
    return FrequencyController(tb, SupportParams(), RecoveryParams(), EventSchedule(),
                              wind_ms=9.0, omega_setpoint_rads=0.68, P_baseline_pre_W=6e6)


def test_support_injects_power_on_underfrequency():
    c = _ctl()
    c.mode = MODE_SUPPORT
    dP, tau, rate = c.dP_target_and_dynamics(df_pu=-0.004, rocof_meas_hz_s=-0.5, omega=0.68)
    assert dP > 0                       # under-frequency -> inject power
    assert dP <= c.sp.dP_max + 1e-9     # respects the magnitude clip


def test_reservoir_guard_truncates_near_floor():
    c = _ctl()
    c.mode = MODE_SUPPORT
    omega_near_floor = IEA15MW.omega_min_rads + 0.01
    dP, _, _ = c.dP_target_and_dynamics(df_pu=-0.01, rocof_meas_hz_s=-0.5,
                                        omega=omega_near_floor)
    assert dP <= 0.0                    # undeliverable support is truncated, not honoured


def test_recovery_underproduces_when_below_setpoint():
    c = _ctl()
    c.mode = MODE_RECOVERY
    dP, _, _ = c.dP_target_and_dynamics(df_pu=-0.002, rocof_meas_hz_s=0.0, omega=0.60)
    assert dP < 0                       # under-produce so the rotor re-accelerates


def test_incumbent_recovery_is_faster():
    inc = RecoveryParams.incumbent()
    shaped = RecoveryParams()
    assert inc.rate_rec_pu_s > shaped.rate_rec_pu_s
    assert inc.tau_rec_s < shaped.tau_rec_s


def test_deadband_ignores_tiny_deviation():
    c = _ctl()
    c.mode = MODE_SUPPORT
    dP, _, _ = c.dP_target_and_dynamics(df_pu=-0.0001, rocof_meas_hz_s=0.0, omega=0.68)
    assert abs(dP) < 1e-6               # inside the 15 mHz deadband
