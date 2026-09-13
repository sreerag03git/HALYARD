r"""Coupled time-domain simulation — the whole causation chain in one state vector (§5).

State y = [ df, pgov, rocof_meas | omega | dP_ctrl | x, x_dot, theta, theta_dot ]
          \-------- grid --------/  rotor    ctrl    \------ platform ------/

grid Delta f, rotor omega, the control command, and the platform DOFs advance on the
same clock, so every feedback loop is genuine: the aggregate wind support feeds the grid
(secondary dip), the rotor-speed change moves thrust, thrust moves the platform, and the
platform's fore-aft velocity feeds back into thrust (aero damping). Integrated with a
fixed-step RK4 (deterministic wall-clock; robust to the controller's clips and mode
switching, which are updated once per step to keep each step's RHS smooth). A warm-up
transient is discarded before any fatigue counting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models.iea15mw import Turbine, IEA15MW
from models.volturnus_s import ReducedPlatformModel, build_reduced_model
from physics import aero, grid as gridmod, platform as plat
from physics.controller import (EventSchedule, FrequencyController, RecoveryParams,
                                SupportParams, MODE_SUPPORT, MODE_RECOVERY)
from physics.grid import GridModel, load_participation
from physics.waves import WaveField

# Full-state indices.
S_DF, S_PGOV, S_ROCOF = 0, 1, 2
S_OMEGA = 3
S_DPCTRL = 4
S_X, S_XDOT, S_THETA, S_THETADOT = 5, 6, 7, 8
N_STATES = 9


@dataclass
class SimConfig:
    wind_ms: float = 9.0
    Hs_m: float = 2.5
    Tp_s: float = 8.0
    wave_seed: int = 1234
    t_end_s: float = 350.0
    dt_s: float = 0.025
    t_discard_s: float = 100.0          # warm-up discarded before fatigue counting
    p_load_pu: float = 0.10             # generation-loss step (system pu)
    wind_capacity_MW: float = 900.0     # fleet capacity providing identical support
    enable_support: bool = True         # False -> wave-only baseline (same seed)
    beta_deg: float = 0.0               # blade pitch (0 for <= rated wind)
    grid: GridModel = field(default_factory=GridModel)
    support: SupportParams = field(default_factory=SupportParams)
    recovery: RecoveryParams = field(default_factory=RecoveryParams)
    schedule: EventSchedule = field(default_factory=EventSchedule)


@dataclass
class SimResult:
    t: np.ndarray
    freq_hz: np.ndarray
    rocof_hz_s: np.ndarray
    omega_rads: np.ndarray
    dP_ctrl_pu: np.ndarray
    P_elec_W: np.ndarray
    thrust_N: np.ndarray
    surge_m: np.ndarray
    pitch_deg: np.ndarray
    surge_vel: np.ndarray
    pitch_rate: np.ndarray
    mode: np.ndarray
    ke_MJ: np.ndarray
    config: SimConfig
    meta: dict
    fidelity: str = "reduced-order (live)"

    @property
    def counting_mask(self) -> np.ndarray:
        return self.t >= self.config.t_discard_s


def run_simulation(cfg: SimConfig, turbine: Turbine | None = None,
                   model: ReducedPlatformModel | None = None) -> SimResult:
    tb = turbine or IEA15MW
    model = model or build_reduced_model()
    wave = WaveField(Hs_m=cfg.Hs_m, Tp_s=cfg.Tp_s, seed=cfg.wave_seed)

    # Precompute wave forces on the step grid; the RK4 midpoint uses the neighbour
    # average (wave force is smooth over one dt, so this is accurate and avoids both
    # per-call interpolation and an over-fine precompute).
    n = int(round(cfg.t_end_s / cfg.dt_s)) + 1
    ts = np.linspace(0.0, cfg.t_end_s, n)
    Fw_surge, Fw_pitch = plat.wave_forces_at(model.platform, wave, ts)

    # Steady operating point and pre-event baseline power.
    omega0, P0 = aero.steady_operating_point(tb, cfg.wind_ms, cfg.beta_deg)
    ctl = FrequencyController(tb, cfg.support, cfg.recovery, cfg.schedule,
                              cfg.wind_ms, omega0, P0)

    participation = load_participation(cfg.wind_capacity_MW, cfg.grid)
    Minv = np.linalg.inv(model.M)
    B, C = model.B, model.C
    z_hub = model.z_hub_m

    # Fast 1-D aero lookups at the fixed blade pitch (hot-loop optimization; equivalent
    # to the 2-D surface at this beta). np.interp is C-fast vs the Python 2-D interp.
    _R = tb.rotor_radius_m
    _half_rho_A = 0.5 * 1.225 * tb.rotor_area_m2
    _lam_tab = np.linspace(0.3, 20.0, 2000)
    _ct_tab = np.array([tb.aero.ct(l, cfg.beta_deg) for l in _lam_tab])
    _cp_tab = np.array([tb.aero.cp(l, cfg.beta_deg) for l in _lam_tab])
    _eta = tb.converter_efficiency
    _J = tb.rotor_inertia_kgm2

    def _thrust(omega, v_nac):
        v_rel = max(cfg.wind_ms - v_nac, 0.1)
        lam = omega * _R / v_rel
        return _half_rho_A * np.interp(lam, _lam_tab, _ct_tab) * v_rel ** 2

    def _domega(omega, v_nac, P_elec):
        v_rel = max(cfg.wind_ms - v_nac, 0.1)
        omega = max(omega, 1e-3)
        lam = omega * _R / v_rel
        T_aero = _half_rho_A * np.interp(lam, _lam_tab, _cp_tab) * v_rel ** 3 / omega
        T_gen = P_elec / (_eta * omega)
        return (T_aero - T_gen) / _J

    # Initial platform state: static mean lean under steady thrust (avoids a startup jolt).
    F_thrust0 = aero.thrust_N(tb, omega0, cfg.wind_ms, cfg.beta_deg, 0.0)
    x0, th0 = model.static_response(F_thrust0)

    y = np.zeros(N_STATES)
    y[S_OMEGA] = omega0
    y[S_X] = x0
    y[S_THETA] = th0

    # Scalar constants for the allocation-free hot loop.
    m00, m01, m10, m11 = Minv[0, 0], Minv[0, 1], Minv[1, 0], Minv[1, 1]
    B00, B01, B10, B11 = B[0, 0], B[0, 1], B[1, 0], B[1, 1]
    C00, C01, C10, C11 = C[0, 0], C[0, 1], C[1, 0], C[1, 1]
    g = cfg.grid
    g2H, gD, gR, gTg, gRes, gTr, gf0 = (2.0 * g.H_sys_s, g.D_load, g.R_sys, g.T_gov_s,
                                        g.reserve_pu, g.T_rocof_s, g.f0_Hz)
    tau_sup = cfg.support.tau_s
    _P_pre, _rated = ctl.P_pre, tb.rated_power_W
    t_event = cfg.schedule.t_event_s
    enable = cfg.enable_support

    def rhs(t: float, y: np.ndarray, fw_s: float, fw_p: float) -> np.ndarray:
        df, pgov, rocof_meas = y[S_DF], y[S_PGOV], y[S_ROCOF]
        omega = y[S_OMEGA]
        dP_ctrl = y[S_DPCTRL]
        x, xdot, theta, thdot = y[S_X], y[S_XDOT], y[S_THETA], y[S_THETADOT]

        v_nac = xdot + z_hub * thdot
        P_elec = ctl.elec_power_W(omega, dP_ctrl)
        F_thrust = _thrust(omega, v_nac)

        # Grid (inlined scalar SFR): aggregate wind support + the load-loss step.
        if enable:
            p_wind = participation * (P_elec - _P_pre) / _rated
            p_load = cfg.p_load_pu if t >= t_event else 0.0
        else:
            p_wind = 0.0
            p_load = 0.0
        ddf = (pgov + p_wind - p_load - gD * df) / g2H
        rocof_true = gf0 * ddf
        dpgov = (-(1.0 / gR) * df - pgov) / gTg
        if (pgov >= gRes and dpgov > 0) or (pgov <= -gRes and dpgov < 0):
            dpgov = 0.0
        drocof = (rocof_true - rocof_meas) / gTr

        d = np.empty(N_STATES)
        d[S_DF] = ddf
        d[S_PGOV] = dpgov
        d[S_ROCOF] = drocof
        d[S_OMEGA] = _domega(omega, v_nac, P_elec)
        d[S_DPCTRL] = (ctl.ddP_ctrl(dP_ctrl, df, rocof_meas, omega)
                       if enable else -dP_ctrl / tau_sup)
        # Platform 2-DOF, scalar: acc = Minv (F_ext - B qdot - C q).
        rx = F_thrust + fw_s - (B00 * xdot + B01 * thdot) - (C00 * x + C01 * theta)
        rt = F_thrust * z_hub + fw_p - (B10 * xdot + B11 * thdot) - (C10 * x + C11 * theta)
        d[S_X] = xdot
        d[S_XDOT] = m00 * rx + m01 * rt
        d[S_THETA] = thdot
        d[S_THETADOT] = m10 * rx + m11 * rt
        return d

    # Fixed-step RK4 with per-step mode update. Wave forces indexed on the half-step grid:
    # step i uses t_half[2(i-1)] (t), [2(i-1)+1] (t+dt/2, twice), [2i] (t+dt).
    dt = cfg.dt_s
    Y = np.empty((n, N_STATES))
    modes = np.empty(n, dtype=int)
    Y[0] = y
    for i in range(1, n):
        t = ts[i - 1]
        fs1, fp1 = Fw_surge[i - 1], Fw_pitch[i - 1]
        fs4, fp4 = Fw_surge[i], Fw_pitch[i]
        fs2, fp2 = 0.5 * (fs1 + fs4), 0.5 * (fp1 + fp4)
        if enable:
            ctl.update_mode(t, y[S_DF], y[S_OMEGA])
        modes[i - 1] = ctl.mode
        k1 = rhs(t, y, fs1, fp1)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1, fs2, fp2)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2, fs2, fp2)
        k4 = rhs(t + dt, y + dt * k3, fs4, fp4)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[i] = y
    modes[-1] = ctl.mode

    # Derived signals.
    omega = Y[:, S_OMEGA]
    dP = Y[:, S_DPCTRL]
    P_elec = np.array([ctl.elec_power_W(w, d) for w, d in zip(omega, dP)])
    v_nac = Y[:, S_XDOT] + z_hub * Y[:, S_THETADOT]
    thrust = np.array([_thrust(w, v) for w, v in zip(omega, v_nac)])
    return SimResult(
        t=ts,
        freq_hz=cfg.grid.f0_Hz * (1.0 + Y[:, S_DF]),
        rocof_hz_s=Y[:, S_ROCOF],
        omega_rads=omega,
        dP_ctrl_pu=dP,
        P_elec_W=P_elec,
        thrust_N=thrust,
        surge_m=Y[:, S_X],
        pitch_deg=np.degrees(Y[:, S_THETA]),
        surge_vel=Y[:, S_XDOT],
        pitch_rate=Y[:, S_THETADOT],
        mode=modes,
        ke_MJ=0.5 * tb.rotor_inertia_kgm2 * omega ** 2 / 1e6,
        config=cfg,
        meta={
            "omega0_rads": omega0, "P0_W": P0, "participation": participation,
            "static_surge_m": x0, "static_pitch_deg": np.degrees(th0),
            "aero_provenance": tb.aero.provenance,
            "wave_Hs_realized": wave.Hs_realized,
        },
    )
