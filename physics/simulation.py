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
    t_end_s: float = 700.0
    dt_s: float = 0.02
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

    def rhs(t: float, y: np.ndarray, F_wave: np.ndarray) -> np.ndarray:
        df, pgov, rocof_meas = y[S_DF], y[S_PGOV], y[S_ROCOF]
        omega = y[S_OMEGA]
        dP_ctrl = y[S_DPCTRL]
        x, xdot, theta, thdot = y[S_X], y[S_XDOT], y[S_THETA], y[S_THETADOT]

        v_nac = xdot + z_hub * thdot
        P_elec = ctl.elec_power_W(omega, dP_ctrl)
        F_thrust = _thrust(omega, v_nac)

        # Grid: aggregate wind support (only if enabled) + the load-loss step.
        if cfg.enable_support:
            p_wind = participation * ctl.delta_wind_pu(omega, dP_ctrl)
            p_load = cfg.p_load_pu if t >= cfg.schedule.t_event_s else 0.0
        else:
            p_wind = 0.0
            p_load = 0.0
        dgrid, _ = cfg.grid.rhs(y[S_DF:S_ROCOF + 1], p_wind, p_load)

        d = np.empty(N_STATES)
        d[S_DF:S_ROCOF + 1] = dgrid
        d[S_OMEGA] = _domega(omega, v_nac, P_elec)
        d[S_DPCTRL] = (ctl.ddP_ctrl(dP_ctrl, df, rocof_meas, omega)
                       if cfg.enable_support else -dP_ctrl / cfg.support.tau_s)
        q = np.array([x, theta])
        qdot = np.array([xdot, thdot])
        F_ext = np.array([F_thrust, F_thrust * z_hub]) + F_wave
        acc = Minv @ (F_ext - B @ qdot - C @ q)
        d[S_X] = xdot
        d[S_XDOT] = acc[0]
        d[S_THETA] = thdot
        d[S_THETADOT] = acc[1]
        return d

    # Fixed-step RK4 with per-step mode update. Wave forces indexed on the half-step grid:
    # step i uses t_half[2(i-1)] (t), [2(i-1)+1] (t+dt/2, twice), [2i] (t+dt).
    dt = cfg.dt_s
    Y = np.empty((n, N_STATES))
    modes = np.empty(n, dtype=int)
    Y[0] = y
    for i in range(1, n):
        t = ts[i - 1]
        Fw1 = np.array([Fw_surge[i - 1], Fw_pitch[i - 1]])
        Fw4 = np.array([Fw_surge[i], Fw_pitch[i]])
        Fw2 = 0.5 * (Fw1 + Fw4)
        if cfg.enable_support:
            ctl.update_mode(t, y[S_DF], y[S_OMEGA])
        modes[i - 1] = ctl.mode
        k1 = rhs(t, y, Fw1)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1, Fw2)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2, Fw2)
        k4 = rhs(t + dt, y + dt * k3, Fw4)
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
