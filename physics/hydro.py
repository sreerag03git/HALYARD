"""6-DOF floating-platform hydrodynamics — Cummins time-domain model (industry standard).

Uses the potential-flow BEM database (``models/data/volturnus_hydro.npz``, computed offline
by Capytaine from the VolturnUS-S hull) to build a time-domain Cummins equation:

    (M + A_inf) q'' + integral_0^t K(t-tau) q'(tau) dtau + B_visc q' + C q
        = F_wave1(t) + F_wave2(t) + F_moor(q) + F_ext(t)

with q the 6 rigid-body DOFs [surge, sway, heave, roll, pitch, yaw]. Terms:
  * A_inf     infinite-frequency added mass (from the BEM A(omega) at high omega);
  * K(t)      radiation retardation (memory) matrix, K(t) = (2/pi) int B(omega) cos(omega t) domega;
  * C         hydrostatic restoring (analytic, from waterplane + buoyancy/gravity) + mooring;
  * F_wave1   first-order wave excitation from the BEM excitation RAOs X(omega, heading);
  * F_wave2   second-order slow-drift (Newman approximation) — see EXTENSIONS.md;
  * F_ext     external loads (rotor thrust at the hub, applied to surge/heave/pitch).

The radiation memory is the frequency-dependence done right (frequency-dependent added mass
AND damping in one convolution), versus constant coefficients. The retardation is evaluated
on a coarse lag grid and convolved with the stored velocity history (vectorized).

BEM data is NOT a runtime dependency (Capytaine runs offline); the live engine only reads the
bundled .npz and integrates. See ``scripts/build_hydro_database.py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from physics.constants import G, RHO_WATER
from physics.waves import WaveField

_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "data")
_HYDRO_NPZ = os.path.join(_DATA, "volturnus_hydro.npz")

DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")

# Cap the BEM frequency range used: above ~1.2 rad/s irregular frequencies (without a free-
# surface lid) corrupt the potential-flow data with unphysical negative damping spikes. The
# wave band (Tp 5-20 s -> omega 0.3-1.3) is covered; radiation damping is taken to 0 beyond,
# which is physical as omega grows. Diagonal damping is also clipped to >= 0 (must be for a
# passive body). A lid re-run in scripts/build_hydro_database.py is the fuller fix (EXTENSIONS).
OMEGA_MAX_USE = 1.2


@dataclass
class HydroDB:
    omega: np.ndarray               # [n_omega] rad/s
    headings: np.ndarray            # [n_head] rad
    A: np.ndarray                   # [n_omega, 6, 6] added mass
    B: np.ndarray                   # [n_omega, 6, 6] radiation damping
    X: np.ndarray                   # [n_omega, n_head, 6] complex excitation-force RAOs
    dof_order: list

    @staticmethod
    def load() -> "HydroDB":
        d = np.load(_HYDRO_NPZ, allow_pickle=True)
        return HydroDB(omega=d["omega"], headings=d["headings"], A=d["added_mass"],
                       B=d["radiation_damping"], X=d["excitation_force"],
                       dof_order=[str(s).lower() for s in d["dof_order"]])

    def A_inf(self) -> np.ndarray:
        """Infinite-frequency added mass ~ A at the highest reliable frequency."""
        mask = self.omega <= OMEGA_MAX_USE
        i = np.where(mask)[0][-1]
        return self.A[i].copy()

    def retardation(self, t_lags: np.ndarray) -> np.ndarray:
        """Radiation retardation K(t) [n_lags, 6, 6] = (2/pi) int_0^wmax B(w) cos(w t) dw."""
        mask = self.omega <= OMEGA_MAX_USE
        w = self.omega[mask]
        B = self.B[mask].copy()                           # [nw, 6, 6]
        # Passivity: diagonal radiation damping must be >= 0 (clip numerical negatives).
        for i in range(6):
            B[:, i, i] = np.maximum(B[:, i, i], 0.0)
        K = np.zeros((len(t_lags), 6, 6))
        for k, t in enumerate(t_lags):
            integrand = B * np.cos(w * t)[:, None, None]  # [nw,6,6]
            K[k] = (2.0 / np.pi) * np.trapezoid(integrand, w, axis=0)
        return K

    def excitation_at(self, heading_rad: float) -> tuple[np.ndarray, np.ndarray]:
        """Complex excitation-force RAOs interpolated to a heading: (omega[used], X[nw,6])."""
        mask = self.omega <= OMEGA_MAX_USE
        w = self.omega[mask]
        # nearest heading (symmetry covers the rest); linear interp between the two nearest.
        h = self.headings
        j = int(np.clip(np.searchsorted(h, heading_rad) - 1, 0, len(h) - 2))
        t = 0.0 if h[j + 1] == h[j] else (heading_rad - h[j]) / (h[j + 1] - h[j])
        X = (1 - t) * self.X[mask, j, :] + t * self.X[mask, j + 1, :]
        return w, X


# ---------------------------------------------------------------------------
# Full-system mass properties (platform + tower + RNA) about the MSL origin.
# Component masses/COGs from the published definitions (see ASSUMPTIONS.md); the moments of
# inertia are built from the component masses and heights (documented estimates).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MassProps:
    m_platform: float = 17.839e6
    z_platform: float = -14.0
    m_tower: float = 1.263e6
    z_tower: float = 60.0
    m_rna: float = 1.017e6
    z_rna: float = 150.0
    r_gyr_platform: float = 30.0     # platform radius of gyration (pitch/roll) [m]
    r_gyr_yaw: float = 35.0          # system radius of gyration (yaw) [m]

    @property
    def m_total(self):
        return self.m_platform + self.m_tower + self.m_rna

    @property
    def z_cog(self):
        return (self.m_platform * self.z_platform + self.m_tower * self.z_tower
                + self.m_rna * self.z_rna) / self.m_total

    def mass_matrix(self) -> np.ndarray:
        """6x6 rigid-body mass matrix about the MSL origin (0,0,0)."""
        m, zg = self.m_total, self.z_cog
        # Pitch/roll inertia about origin: parallel-axis of point masses + platform own inertia.
        I_pts = (self.m_platform * self.z_platform ** 2 + self.m_tower * self.z_tower ** 2
                 + self.m_rna * self.z_rna ** 2)
        I_own = self.m_platform * self.r_gyr_platform ** 2 \
            + self.m_tower * (130.0 ** 2) / 12.0
        I_pitch = I_pts + I_own
        I_yaw = m * self.r_gyr_yaw ** 2
        M = np.zeros((6, 6))
        M[0, 0] = M[1, 1] = M[2, 2] = m
        M[0, 4] = M[4, 0] = m * zg          # surge-pitch coupling (COG above origin)
        M[1, 3] = M[3, 1] = -m * zg         # sway-roll coupling
        M[3, 3] = I_pitch                   # roll
        M[4, 4] = I_pitch                   # pitch
        M[5, 5] = I_yaw                     # yaw
        return M


def hydrostatic_stiffness_6dof(platform) -> np.ndarray:
    """Analytic hydrostatic restoring [6x6] about the MSL origin.

    Heave: rho g A_wp. Roll/pitch: rho g I_wp + W (z_B - z_G). Others ~ 0. This is more
    reliable than mesh hydrostatics for this configuration and is documented in ASSUMPTIONS.md.
    """
    from models.volturnus_s import VOLTURNUS_S
    p = platform or VOLTURNUS_S
    A_wp = p.waterplane_area_m2
    D = p.column_diameter_m
    r = p.column_spacing_m
    # Waterplane second moment about a horizontal axis through the centre: own + parallel axis
    # of the 3 offset columns (sum of x_i^2 = 1.5 r^2 for 3 symmetric points).
    A_col = np.pi / 4 * D ** 2
    I_own = 4 * (np.pi / 64 * D ** 4)
    I_wp = I_own + A_col * 1.5 * r ** 2
    C = np.zeros((6, 6))
    C[2, 2] = RHO_WATER * G * A_wp
    return C, I_wp


def buoyancy_center_z(platform) -> float:
    """Approx centre of buoyancy z from the column + pontoon volumes."""
    from models.volturnus_s import VOLTURNUS_S
    p = platform or VOLTURNUS_S
    A_col = np.pi / 4 * p.column_diameter_m ** 2
    V_col = A_col * p.draft_m
    z_col = -p.draft_m / 2
    length = p.column_spacing_m - p.column_diameter_m
    V_pon = length * 12.5 * 7.0
    z_pon = -p.draft_m + 7.0 / 2
    Vtot = 4 * V_col + 3 * V_pon
    return (4 * V_col * z_col + 3 * V_pon * z_pon) / Vtot


@dataclass
class Mooring:
    """Linearized 3-line catenary mooring about the operating point (fitted to periods)."""
    k_horizontal: float = 9.3e4      # surge/sway stiffness [N/m] (fitted to surge period ~120 s)
    k_yaw: float = 2.8e8             # yaw stiffness [N m/rad] (fitted to yaw period ~85 s)
    k_heave: float = 6.0e4           # small vertical stiffness from line geometry [N/m]
    fairlead_z: float = -14.0        # fairlead depth [m] (surge<->pitch coupling)
    pitch_extra: float = 0.52e9      # extra roll/pitch restoring from lines (fitted, ~28 s)

    def stiffness(self) -> np.ndarray:
        C = np.zeros((6, 6))
        kh, zf = self.k_horizontal, self.fairlead_z
        C[0, 0] = C[1, 1] = kh
        C[2, 2] = self.k_heave
        C[5, 5] = self.k_yaw
        # Horizontal load at fairlead depth couples surge<->pitch and sway<->roll.
        C[0, 4] = C[4, 0] = kh * zf
        C[1, 3] = C[3, 1] = -kh * zf
        C[3, 3] += kh * zf ** 2 + self.pitch_extra
        C[4, 4] += kh * zf ** 2 + self.pitch_extra
        return C


@dataclass
class Hydro6DOF:
    M: np.ndarray                    # rigid-body mass [6x6]
    A_inf: np.ndarray                # infinite-frequency added mass [6x6]
    C: np.ndarray                    # total restoring (hydrostatic + mooring) [6x6]
    B_visc: np.ndarray               # linear viscous damping [6x6]
    K_lags: np.ndarray               # retardation lag times [n_lag]
    K: np.ndarray                    # retardation kernel [n_lag, 6, 6]
    db: HydroDB
    Minv: np.ndarray = field(default=None)

    def __post_init__(self):
        self.Minv = np.linalg.inv(self.M + self.A_inf)

    def natural_periods(self) -> dict:
        """Undamped natural periods per DOF, iterating the frequency-dependent added mass."""
        out = {}
        Ainterp = lambda w, i: np.interp(  # noqa: E731
            w, self.db.omega, self.db.A[:, i, i])
        for i, name in enumerate(DOF_NAMES):
            c = self.C[i, i]
            if c <= 0:
                out[name] = np.inf
                continue
            wn = np.sqrt(c / (self.M[i, i] + self.A_inf[i, i]))
            for _ in range(30):
                a = Ainterp(min(wn, self.db.omega.max()), i)
                wn = np.sqrt(c / (self.M[i, i] + a))
            out[name] = 2 * np.pi / wn
        return out

    def radiation_force(self, vel_hist: np.ndarray, dt: float) -> np.ndarray:
        """Convolve the stored velocity history with the retardation kernel: f_rad = -int K v.

        ``vel_hist`` is [n, 6] most-recent-last at the sim step dt. Vectorized over lags.
        """
        n = len(vel_hist)
        idx = (n - 1 - np.round(self.K_lags / dt)).astype(int)
        valid = idx >= 0
        idx = idx[valid]
        Kv = self.K[valid]                          # [nlag,6,6]
        v = vel_hist[idx]                           # [nlag,6]
        # trapezoidal weights over the (uniform) lag grid
        dlag = self.K_lags[1] - self.K_lags[0] if len(self.K_lags) > 1 else 1.0
        f = np.einsum("kij,kj->i", Kv, v) * dlag
        return -f


def build_hydro_6dof(mass: MassProps | None = None, mooring: Mooring | None = None,
                     platform=None, mem_T: float = 30.0, mem_dt: float = 0.1) -> Hydro6DOF:
    from models.volturnus_s import VOLTURNUS_S
    mass = mass or MassProps()
    mooring = mooring or Mooring()
    platform = platform or VOLTURNUS_S
    db = HydroDB.load()
    M = mass.mass_matrix()
    A_inf = db.A_inf()
    C_hydro, I_wp = hydrostatic_stiffness_6dof(platform)
    W = mass.m_total * G
    z_B = buoyancy_center_z(platform)
    c_roll_pitch = RHO_WATER * G * I_wp + W * (z_B - mass.z_cog)
    C_hydro[3, 3] += c_roll_pitch
    C_hydro[4, 4] += c_roll_pitch
    C = C_hydro + mooring.stiffness()
    # Light linear viscous damping (Morison drag linearization); refined in EXTENSIONS.
    B_visc = np.diag([2e5, 2e5, 8e5, 4e8, 4e8, 2e8]).astype(float)
    lags = np.arange(0.0, mem_T + mem_dt, mem_dt)
    K = db.retardation(lags)
    return Hydro6DOF(M=M, A_inf=A_inf, C=C, B_visc=B_visc, K_lags=lags, K=K, db=db)


def wave_excitation_6dof(db: HydroDB, wave: WaveField, t: np.ndarray,
                         heading_rad: float = 0.0) -> np.ndarray:
    """First-order 6-DOF wave-excitation force time series [n_t, 6] from the BEM RAOs."""
    w_used, X = db.excitation_at(heading_rad)          # [nw_used, 6] complex
    # Interpolate the complex RAO to each JONSWAP component frequency.
    Xr = np.array([np.interp(wave.w, w_used, X[:, d].real) for d in range(6)]).T  # [ncomp,6]
    Xi = np.array([np.interp(wave.w, w_used, X[:, d].imag) for d in range(6)]).T
    amp = wave.amp[:, None]                             # [ncomp,1]
    ph = wave.phase[:, None]
    F = np.zeros((len(t), 6))
    WT = np.outer(t, wave.w)                            # [nt, ncomp]
    cos = np.cos(WT + ph.T)
    sin = np.sin(WT + ph.T)
    for d in range(6):
        # Re{ amp * (Xr+iXi) * e^{i(wt+ph)} } = amp[ Xr cos - Xi sin ]
        F[:, d] = (amp[:, 0] * (cos * Xr[:, d] - sin * Xi[:, d])).sum(axis=1)
    return F
