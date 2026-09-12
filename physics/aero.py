"""Electrical-to-structural coupling — aerodynamics and rotor dynamics (§5.4).

This is the hinge of the whole argument: a change in rotor speed changes the tip-speed
ratio -> the thrust coefficient -> the horizontal rotor force that pushes the floating
platform. Break this link and there is no mechanism.

    lambda   = omega R / V_rel                        tip-speed ratio [-]
    F_thrust = 1/2 rho_air A C_T(lambda,beta) V_rel^2  rotor thrust [N]
    T_aero   = 1/2 rho_air A C_P(lambda,beta) V_rel^3 / omega   aero torque [N m]
    V_rel    = V - x_dot_nacelle                       relative wind incl. platform motion

Rotor ODE (the KE reservoir):
    J domega/dt = T_aero - T_gen ,   T_gen = P_elec / (eta_conv omega)

The nacelle fore-aft velocity x_dot_nacelle = x_dot_surge + z_hub * theta_dot couples
the platform motion back into thrust — the source of aerodynamic (positive OR negative)
damping, essential for the platform stability check (§5.5). We use V_rel consistently in
both lambda and the dynamic pressure (standard reduced-order practice).
"""
from __future__ import annotations

import numpy as np

from models.iea15mw import Turbine
from physics.constants import RHO_AIR


def nacelle_fore_aft_velocity(x_dot_surge: float, theta_dot: float, z_hub: float) -> float:
    """Fore-aft velocity of the hub from surge + pitch rotation [m/s]."""
    return x_dot_surge + z_hub * theta_dot


def relative_wind(V: float, x_dot_nacelle: float) -> float:
    """Relative wind at the rotor [m/s], floored to a small positive value."""
    return max(V - x_dot_nacelle, 0.1)


def thrust_N(turbine: Turbine, omega: float, V: float, beta_deg: float,
             x_dot_nacelle: float = 0.0) -> float:
    V_rel = relative_wind(V, x_dot_nacelle)
    lam = omega * turbine.rotor_radius_m / V_rel
    ct = turbine.aero.ct(lam, beta_deg)
    return 0.5 * RHO_AIR * turbine.rotor_area_m2 * ct * V_rel ** 2


def aero_torque_Nm(turbine: Turbine, omega: float, V: float, beta_deg: float,
                   x_dot_nacelle: float = 0.0) -> float:
    V_rel = relative_wind(V, x_dot_nacelle)
    omega = max(omega, 1e-3)
    lam = omega * turbine.rotor_radius_m / V_rel
    cp = turbine.aero.cp(lam, beta_deg)
    return 0.5 * RHO_AIR * turbine.rotor_area_m2 * cp * V_rel ** 3 / omega


def aero_power_W(turbine: Turbine, omega: float, V: float, beta_deg: float,
                 x_dot_nacelle: float = 0.0) -> float:
    V_rel = relative_wind(V, x_dot_nacelle)
    lam = omega * turbine.rotor_radius_m / V_rel
    cp = turbine.aero.cp(lam, beta_deg)
    return 0.5 * RHO_AIR * turbine.rotor_area_m2 * cp * V_rel ** 3


def generator_torque_Nm(turbine: Turbine, P_elec_W: float, omega: float) -> float:
    """T_gen = P_elec / (eta_conv omega) [N m]."""
    return P_elec_W / (turbine.converter_efficiency * max(omega, 1e-3))


def rotor_domega_dt(turbine: Turbine, omega: float, V: float, beta_deg: float,
                    P_elec_W: float, x_dot_nacelle: float = 0.0) -> float:
    """J domega/dt = T_aero - T_gen  ->  domega/dt [rad/s^2]."""
    T_aero = aero_torque_Nm(turbine, omega, V, beta_deg, x_dot_nacelle)
    T_gen = generator_torque_Nm(turbine, P_elec_W, omega)
    return (T_aero - T_gen) / turbine.rotor_inertia_kgm2


def steady_operating_point(turbine: Turbine, V: float, beta_deg: float = 0.0
                           ) -> tuple[float, float]:
    """Solve the steady rotor speed for wind V under the MPPT/rated baseline.

    Returns (omega0 [rad/s], P_elec0 [W]). Below rated: MPPT tracks lambda_opt (capped
    to the speed envelope). At/above rated wind: omega = omega_rated, P = rated.
    """
    lam_opt, _, cp_max = turbine.aero.optimal()
    rho = RHO_AIR
    R = turbine.rotor_radius_m
    A = turbine.rotor_area_m2
    K_opt = 0.5 * rho * A * R ** 3 * cp_max / lam_opt ** 3
    # MPPT equilibrium: T_aero(omega,V) = K_opt omega^2. Solve by bisection on omega.
    lo, hi = turbine.omega_min_rads, turbine.omega_max_rads

    def resid(w):
        return aero_torque_Nm(turbine, w, V, beta_deg) - K_opt * w ** 2

    # If even at omega_max the aero torque exceeds the MPPT demand, we are at/above
    # rated -> clamp to rated speed and rated power.
    if resid(hi) > 0:
        omega0 = turbine.omega_rated_rads
        return omega0, turbine.rated_power_W
    if resid(lo) < 0:
        omega0 = turbine.omega_min_rads
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if resid(mid) > 0:
                lo = mid
            else:
                hi = mid
        omega0 = 0.5 * (lo + hi)
    P0 = min(turbine.converter_efficiency * K_opt * omega0 ** 3, turbine.rated_power_W)
    return omega0, P0
