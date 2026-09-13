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
    hydro_6dof: bool = True             # True: BEM 6-DOF Cummins hydro; False: fitted 2-DOF
    wave_heading_deg: float = 0.0       # wave direction (0 = aligned with wind/+x)
    turbulence_TI: float = 0.0          # >0 enables Kaimal turbulent inflow at this intensity
    blade_pitch: bool = True            # ROSCO-style collective pitch (enables above-rated)
    dynamic_inflow: bool = True         # first-order dynamic-inflow lag on rotor thrust
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
    # 6-DOF extras (present when hydro_6dof; zeros/None otherwise).
    heave_m: np.ndarray = None
    roll_deg: np.ndarray = None
    sway_m: np.ndarray = None
    yaw_deg: np.ndarray = None
    heave_vel: np.ndarray = None
    blade_pitch_deg: np.ndarray = None
    wind_ms: np.ndarray = None

    @property
    def counting_mask(self) -> np.ndarray:
        return self.t >= self.config.t_discard_s


def run_simulation(cfg: SimConfig, turbine: Turbine | None = None,
                   model: ReducedPlatformModel | None = None) -> SimResult:
    """Dispatch to the 6-DOF BEM hydro engine (default) or the fitted 2-DOF engine."""
    if cfg.hydro_6dof:
        return _run_6dof(cfg, turbine)
    return _run_2dof(cfg, turbine, model)


def _run_2dof(cfg: SimConfig, turbine: Turbine | None = None,
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


# 6-DOF states: grid(3) | omega | dP_ctrl | q(6) | qdot(6) | beta | I_pitch | F_lag.
H_Q0 = 5           # surge, sway, heave, roll, pitch, yaw
H_V0 = 11          # velocities
S_BETA = 17        # collective blade pitch [rad]
S_IPITCH = 18      # pitch-controller integrator
S_FLAG = 19        # dynamic-inflow lagged thrust [N]
N_STATES_6 = 20
_HYDRO_CACHE = {}


def _get_hydro_model():
    """Cached default 6-DOF hydro model (BEM database + retardation)."""
    from physics.hydro import build_hydro_6dof
    if "m" not in _HYDRO_CACHE:
        _HYDRO_CACHE["m"] = build_hydro_6dof()
    return _HYDRO_CACHE["m"]


def _steady_pitch_deg(tb, omega0, V, P0, interp2, CP2, half_rho_A, R, eta):
    """Collective pitch [deg] that makes aero power match the operating point (0 below rated)."""
    lam = omega0 * R / max(V, 0.1)
    target_Paero = P0 / eta
    best, best_err = 0.0, 1e30
    for beta in np.linspace(0.0, 25.0, 60):
        Paero = half_rho_A * interp2(CP2, lam, beta) * V ** 3
        err = abs(Paero - target_Paero)
        if err < best_err:
            best, best_err = beta, err
    return best


def _run_6dof(cfg: SimConfig, turbine: Turbine | None = None) -> SimResult:
    """Coupled sim with the BEM 6-DOF Cummins platform (industry-standard hydro)."""
    from physics.hydro import wave_excitation_6dof
    tb = turbine or IEA15MW
    hy = _get_hydro_model()
    wave = WaveField(Hs_m=cfg.Hs_m, Tp_s=cfg.Tp_s, seed=cfg.wave_seed)

    n = int(round(cfg.t_end_s / cfg.dt_s)) + 1
    ts = np.linspace(0.0, cfg.t_end_s, n)
    dt = cfg.dt_s
    Fwave = wave_excitation_6dof(hy.db, wave, ts, np.radians(cfg.wave_heading_deg))  # [n,6]

    omega0, P0 = aero.steady_operating_point(tb, cfg.wind_ms, cfg.beta_deg)
    ctl = FrequencyController(tb, cfg.support, cfg.recovery, cfg.schedule,
                              cfg.wind_ms, omega0, P0)
    participation = load_participation(cfg.wind_capacity_MW, cfg.grid)
    z_hub = tb.hub_height_m

    # Wind: steady, or Kaimal turbulent inflow (§5.8) at the requested intensity.
    if cfg.turbulence_TI > 0:
        from physics.wind import TurbulentWind
        V_arr = TurbulentWind(U_mean=cfg.wind_ms, TI=cfg.turbulence_TI, z_hub=z_hub,
                              seed=cfg.wave_seed + 7).series(ts)
    else:
        V_arr = np.full(n, cfg.wind_ms)

    # Fast 2-D aero interpolation over (lambda, beta) from the (real) rotor deck.
    _R = tb.rotor_radius_m
    _half_rho_A = 0.5 * 1.225 * tb.rotor_area_m2
    _lamg = np.linspace(0.3, 18.0, 220)
    _betag = np.linspace(-5.0, 40.0, 91)     # deg
    _CT2 = np.array([[tb.aero.ct(l, b) for b in _betag] for l in _lamg])
    _CP2 = np.array([[tb.aero.cp(l, b) for b in _betag] for l in _lamg])
    _eta, _J = tb.converter_efficiency, tb.rotor_inertia_kgm2
    _dl = _lamg[1] - _lamg[0]
    _db = _betag[1] - _betag[0]

    def _interp2(tab, lam, beta_deg):
        lam = min(max(lam, _lamg[0]), _lamg[-2])
        b = min(max(beta_deg, _betag[0]), _betag[-2])
        i = int((lam - _lamg[0]) / _dl); j = int((b - _betag[0]) / _db)
        tl = (lam - _lamg[i]) / _dl; tb_ = (b - _betag[j]) / _db
        return (tab[i, j] * (1 - tl) * (1 - tb_) + tab[i + 1, j] * tl * (1 - tb_)
                + tab[i, j + 1] * (1 - tl) * tb_ + tab[i + 1, j + 1] * tl * tb_)

    def _thrust(omega, v_nac, beta_deg, V):
        v_rel = max(V - v_nac, 0.1)
        return _half_rho_A * _interp2(_CT2, omega * _R / v_rel, beta_deg) * v_rel ** 2

    def _domega(omega, v_nac, P_elec, beta_deg, V):
        v_rel = max(V - v_nac, 0.1)
        omega = max(omega, 1e-3)
        cp = _interp2(_CP2, omega * _R / v_rel, beta_deg)
        return (_half_rho_A * cp * v_rel ** 3 / omega - P_elec / (_eta * omega)) / _J

    Minv, C, Bv = hy.Minv, hy.C, hy.B_visc
    g = cfg.grid
    g2H, gD, gR, gTg, gRes, gTr = (2 * g.H_sys_s, g.D_load, g.R_sys, g.T_gov_s,
                                   g.reserve_pu, g.T_rocof_s)
    gf0, tau_sup = g.f0_Hz, cfg.support.tau_s
    _P_pre, _rated, t_event, enable = ctl.P_pre, tb.rated_power_W, cfg.schedule.t_event_s, \
        cfg.enable_support

    # ROSCO-style collective-pitch controller (active above rated) + floating feedback.
    om_rated = tb.omega_rated_rads
    pitch_on = cfg.blade_pitch
    KP_P, KI_P = 6.0, 2.0            # PI gains on rotor-speed error (rad per rad/s)
    KF_P = 0.05                      # floating (nacelle-velocity) feedback gain
    BETA_MAX = np.radians(25.0)
    PITCH_RATE = np.radians(8.0)     # actuator rate limit [rad/s]
    TAU_PITCH = 0.10                 # actuator lag [s]
    tau_ind = 4.0 if cfg.dynamic_inflow else 0.02   # dynamic-inflow time constant [s]

    # Steady blade pitch to reach the operating point (above rated needs beta>0).
    beta0_deg = _steady_pitch_deg(tb, omega0, cfg.wind_ms, P0, _interp2, _CP2, _half_rho_A,
                                  _R, _eta)
    F_thrust0 = _thrust(omega0, 0.0, beta0_deg, cfg.wind_ms)
    F0 = np.zeros(6)
    F0[0] = F_thrust0
    F0[4] = F_thrust0 * z_hub
    q0 = np.linalg.solve(C, F0)

    y = np.zeros(N_STATES_6)
    y[S_OMEGA] = omega0
    y[H_Q0:H_Q0 + 6] = q0
    y[S_BETA] = np.radians(beta0_deg)
    y[S_FLAG] = F_thrust0
    vel_hist = np.zeros((n, 6))

    def rhs(t, y, fw, f_rad, V):
        df, pgov, rocof_meas = y[0], y[1], y[2]
        omega, dP_ctrl = y[S_OMEGA], y[S_DPCTRL]
        beta, I_p, F_lag = y[S_BETA], y[S_IPITCH], y[S_FLAG]
        qd = y[H_V0:H_V0 + 6]
        q = y[H_Q0:H_Q0 + 6]
        v_nac = qd[0] + z_hub * qd[4]
        beta_deg = np.degrees(beta)
        P_elec = ctl.elec_power_W(omega, dP_ctrl)
        F_thrust_qs = _thrust(omega, v_nac, beta_deg, V)
        if enable:
            p_wind = participation * (P_elec - _P_pre) / _rated
            p_load = cfg.p_load_pu if t >= t_event else 0.0
        else:
            p_wind = p_load = 0.0
        ddf = (pgov + p_wind - p_load - gD * df) / g2H
        dpgov = (-(1.0 / gR) * df - pgov) / gTg
        if (pgov >= gRes and dpgov > 0) or (pgov <= -gRes and dpgov < 0):
            dpgov = 0.0
        drocof = (gf0 * ddf - rocof_meas) / gTr
        # Pitch controller (PI on omega error above rated + floating feedback).
        e = omega - om_rated
        beta_cmd = KP_P * e + KI_P * I_p + KF_P * v_nac
        beta_cmd = min(max(beta_cmd, 0.0), BETA_MAX)
        saturated = (beta_cmd <= 0.0 and e < 0) or (beta_cmd >= BETA_MAX and e > 0)
        dI = 0.0 if (not pitch_on or saturated) else e
        dbeta = 0.0 if not pitch_on else np.clip((beta_cmd - beta) / TAU_PITCH,
                                                 -PITCH_RATE, PITCH_RATE)
        dF_lag = (F_thrust_qs - F_lag) / tau_ind          # dynamic inflow lag
        d = np.empty(N_STATES_6)
        d[0], d[1], d[2] = ddf, dpgov, drocof
        d[S_OMEGA] = _domega(omega, v_nac, P_elec, beta_deg, V)
        d[S_DPCTRL] = (ctl.ddP_ctrl(dP_ctrl, df, rocof_meas, omega)
                       if enable else -dP_ctrl / tau_sup)
        F_ext = np.zeros(6)
        F_ext[0] = F_lag
        F_ext[4] = F_lag * z_hub
        qdd = Minv @ (F_ext + fw + f_rad - Bv @ qd - C @ q)
        d[H_Q0:H_Q0 + 6] = qd
        d[H_V0:H_V0 + 6] = qdd
        d[S_BETA], d[S_IPITCH], d[S_FLAG] = dbeta, dI, dF_lag
        return d

    Y = np.empty((n, N_STATES_6))
    modes = np.empty(n, dtype=int)
    Y[0] = y
    for i in range(1, n):
        t = ts[i - 1]
        vel_hist[i - 1] = y[H_V0:H_V0 + 6]
        f_rad = hy.radiation_force(vel_hist[:i], dt) if i > 1 else np.zeros(6)
        fw1, fw4 = Fwave[i - 1], Fwave[i]
        fw2 = 0.5 * (fw1 + fw4)
        V1, V4 = V_arr[i - 1], V_arr[i]
        V2 = 0.5 * (V1 + V4)
        if enable:
            ctl.update_mode(t, y[S_DF], y[S_OMEGA])
        modes[i - 1] = ctl.mode
        k1 = rhs(t, y, fw1, f_rad, V1)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1, fw2, f_rad, V2)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2, fw2, f_rad, V2)
        k4 = rhs(t + dt, y + dt * k3, fw4, f_rad, V4)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[i] = y
    modes[-1] = ctl.mode

    omega = Y[:, S_OMEGA]
    dP = Y[:, S_DPCTRL]
    P_elec = np.array([ctl.elec_power_W(w, d) for w, d in zip(omega, dP)])
    v_nac = Y[:, H_V0] + z_hub * Y[:, H_V0 + 4]
    thrust = Y[:, S_FLAG]      # the platform-forcing (dynamic-inflow lagged) thrust
    return SimResult(
        t=ts, freq_hz=gf0 * (1.0 + Y[:, S_DF]), rocof_hz_s=Y[:, S_ROCOF],
        omega_rads=omega, dP_ctrl_pu=dP, P_elec_W=P_elec, thrust_N=thrust,
        surge_m=Y[:, H_Q0 + 0], pitch_deg=np.degrees(Y[:, H_Q0 + 4]),
        surge_vel=Y[:, H_V0 + 0], pitch_rate=Y[:, H_V0 + 4], mode=modes,
        ke_MJ=0.5 * tb.rotor_inertia_kgm2 * omega ** 2 / 1e6, config=cfg,
        heave_m=Y[:, H_Q0 + 2], roll_deg=np.degrees(Y[:, H_Q0 + 3]),
        sway_m=Y[:, H_Q0 + 1], yaw_deg=np.degrees(Y[:, H_Q0 + 5]),
        heave_vel=Y[:, H_V0 + 2], fidelity="reduced-order (live, 6-DOF BEM hydro)",
        blade_pitch_deg=np.degrees(Y[:, S_BETA]), wind_ms=V_arr,
        meta={"omega0_rads": omega0, "P0_W": P0, "participation": participation,
              "static_surge_m": q0[0], "static_pitch_deg": np.degrees(q0[4]),
              "static_heave_m": q0[2], "beta0_deg": beta0_deg,
              "aero_provenance": tb.aero.provenance, "realized_TI": float(np.std(V_arr) /
              np.mean(V_arr)) if cfg.turbulence_TI > 0 else 0.0,
              "wave_Hs_realized": wave.Hs_realized, "hydro": "6-DOF BEM Cummins"})
