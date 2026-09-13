"""Rotor aerodynamic performance surfaces C_P(lambda, beta) and C_T(lambda, beta).

WHAT THIS IS (provenance — read this)
-------------------------------------
The IEA-15-240-RWT rotor-performance deck (``Cp_Ct_Cq.txt`` from the public
OpenFAST model) is the authoritative source. HALYARD will **use that file
verbatim if it is present** at ``models/data/Cp_Ct_Cq.txt`` — drop it in for full
fidelity and this module loads it.

When that file is absent (as in a clean clone), we fall back to **reduced-order
analytical surfaces** built from the well-known empirical Heier/Slootweg C_P(lambda,beta)
form, affine-calibrated in tip-speed-ratio and amplitude to the *published*
IEA-15MW anchor points, plus a shaped C_T surface calibrated to the published
rated thrust. This is a documented reduced-order stand-in, **not** the NREL deck,
and it is labelled as such everywhere it is used (``.provenance``). See
ASSUMPTIONS.md.

Calibration anchors (public IEA-15-240-RWT definition, NREL/TP-5000-75698):
  * C_P peak  ~ 0.489  at  lambda_opt ~ 9.0, blade pitch beta = 0 deg
  * rated wind 10.59 m/s, rated rotor speed 7.56 rpm  ->  lambda ~ 8.95 at rated
  * rated rotor thrust ~ 2.4 MN  ->  C_T ~ 0.79 at the region-2 operating point

Physics
-------
lambda = omega * R / V is the tip-speed ratio; beta is collective blade pitch (deg).
C_P sets aerodynamic torque/power, C_T sets rotor thrust (§5.4). The *local slope*
dC_T/dlambda near the operating point is what converts a control-driven rotor-speed
change into a thrust transient that moves the platform — the mechanism of the whole
study — so the surface is calibrated to have a physically reasonable slope there.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_NREL_DECK = os.path.join(_DATA_DIR, "Cp_Ct_Cq.txt")

# Published IEA-15MW anchors used for calibration of the reduced-order fallback.
CP_MAX_ANCHOR = 0.489
LAMBDA_OPT_ANCHOR = 9.0
CT_REGION2_ANCHOR = 0.79       # gives ~2.4 MN at rated wind (verified in build script)
LAMBDA_CT_PEAK = 10.2          # C_T peaks slightly above lambda_opt (physical)


# ---------------------------------------------------------------------------
# Reduced-order analytical surfaces (fallback)
# ---------------------------------------------------------------------------
def _heier_cp_raw(lam: np.ndarray, beta_deg: np.ndarray) -> np.ndarray:
    """Classic empirical C_P(lambda, beta) (Heier/Slootweg form), uncalibrated.

    Returns the bare surface; ``cp_surface`` rescales it in lambda and amplitude
    to hit the IEA-15MW anchors.
    """
    lam = np.asarray(lam, dtype=float)
    beta = np.asarray(beta_deg, dtype=float)
    inv_li = 1.0 / (lam + 0.08 * beta) - 0.035 / (beta ** 3 + 1.0)
    # Guard against division blow-ups where inv_li -> 0.
    inv_li = np.where(np.abs(inv_li) < 1e-6, 1e-6, inv_li)
    li = 1.0 / inv_li
    cp = 0.5176 * (116.0 / li - 0.4 * beta - 5.0) * np.exp(-21.0 / li) + 0.0068 * lam
    return np.maximum(cp, 0.0)


# Determine the raw form's natural optimum once, at import, to build the affine map.
def _raw_optimum() -> tuple[float, float]:
    lam_grid = np.linspace(0.5, 20.0, 4000)
    cp = _heier_cp_raw(lam_grid, np.zeros_like(lam_grid))
    i = int(np.argmax(cp))
    return float(lam_grid[i]), float(cp[i])


_RAW_LAMBDA_OPT, _RAW_CP_MAX = _raw_optimum()
_LAMBDA_STRETCH = _RAW_LAMBDA_OPT / LAMBDA_OPT_ANCHOR   # maps target lambda -> raw lambda
_CP_AMP = CP_MAX_ANCHOR / _RAW_CP_MAX


def cp_surface(lam, beta_deg=0.0) -> np.ndarray:
    """Reduced-order C_P(lambda, beta) calibrated to IEA-15MW (peak 0.489 @ lambda 9)."""
    lam = np.asarray(lam, dtype=float)
    beta = np.asarray(beta_deg, dtype=float)
    cp = _CP_AMP * _heier_cp_raw(lam * _LAMBDA_STRETCH, beta)
    return np.maximum(cp, 0.0)


def ct_surface(lam, beta_deg=0.0) -> np.ndarray:
    """Reduced-order C_T(lambda, beta), calibrated to rated thrust.

    Shape: C_T rises ~linearly with lambda, peaks at LAMBDA_CT_PEAK, then decays —
    the standard qualitative behaviour — so dC_T/dlambda > 0 through the operating
    region (7 < lambda < 10). Pitch to feather (beta > 0) reduces thrust.
    """
    lam = np.asarray(lam, dtype=float)
    beta = np.asarray(beta_deg, dtype=float)
    r = np.clip(lam, 0.0, None) / LAMBDA_CT_PEAK
    # h(r) = r * exp(1 - r): peaks at r = 1 with value 1, ~linear near 0.
    shape = r * np.exp(1.0 - r)
    # Normalise so shape at the region-2 anchor equals 1, then scale to the anchor C_T.
    r0 = LAMBDA_OPT_ANCHOR / LAMBDA_CT_PEAK
    shape0 = r0 * np.exp(1.0 - r0)
    ct = CT_REGION2_ANCHOR * shape / shape0
    pitch_factor = np.exp(-0.04 * np.maximum(beta, 0.0))   # feathering sheds thrust
    return np.maximum(ct * pitch_factor, 0.0)


# ---------------------------------------------------------------------------
# Table loader with interpolation (uses NREL deck if present, else builds fallback)
# ---------------------------------------------------------------------------
@dataclass
class AeroSurfaces:
    """Bilinear-interpolated C_P / C_T surfaces with explicit provenance."""

    lambda_grid: np.ndarray     # tip-speed ratios [-]
    beta_grid: np.ndarray       # blade pitch angles [deg]
    cp_table: np.ndarray        # shape (n_lambda, n_beta)
    ct_table: np.ndarray        # shape (n_lambda, n_beta)
    provenance: str

    def cp(self, lam: float, beta_deg: float = 0.0) -> float:
        return float(self._interp(self.cp_table, lam, beta_deg))

    def ct(self, lam: float, beta_deg: float = 0.0) -> float:
        return float(self._interp(self.ct_table, lam, beta_deg))

    def _interp(self, table: np.ndarray, lam: float, beta_deg: float) -> float:
        # Clamp to table support (extrapolation off a rotor map is meaningless).
        lam = float(np.clip(lam, self.lambda_grid[0], self.lambda_grid[-1]))
        beta = float(np.clip(beta_deg, self.beta_grid[0], self.beta_grid[-1]))
        i = np.clip(np.searchsorted(self.lambda_grid, lam) - 1, 0, len(self.lambda_grid) - 2)
        j = np.clip(np.searchsorted(self.beta_grid, beta) - 1, 0, len(self.beta_grid) - 2)
        l0, l1 = self.lambda_grid[i], self.lambda_grid[i + 1]
        b0, b1 = self.beta_grid[j], self.beta_grid[j + 1]
        tl = 0.0 if l1 == l0 else (lam - l0) / (l1 - l0)
        tb = 0.0 if b1 == b0 else (beta - b0) / (b1 - b0)
        f00, f01 = table[i, j], table[i, j + 1]
        f10, f11 = table[i + 1, j], table[i + 1, j + 1]
        return (f00 * (1 - tl) * (1 - tb) + f10 * tl * (1 - tb)
                + f01 * (1 - tl) * tb + f11 * tl * tb)

    def optimal(self) -> tuple[float, float, float]:
        """Return (lambda_opt, beta_opt_deg, cp_max) from the table (beta>=0)."""
        mask = self.beta_grid >= 0
        sub = self.cp_table[:, mask]
        i, jj = np.unravel_index(int(np.argmax(sub)), sub.shape)
        return (float(self.lambda_grid[i]), float(self.beta_grid[mask][jj]),
                float(sub[i, jj]))


def load_aero_surfaces() -> AeroSurfaces:
    """Load the NREL rotor deck if present, else build the calibrated fallback."""
    if os.path.exists(_NREL_DECK):
        return _load_nrel_deck(_NREL_DECK)
    return _build_reduced_surfaces()


def _build_reduced_surfaces() -> AeroSurfaces:
    lam = np.linspace(0.5, 18.0, 120)
    beta = np.linspace(0.0, 30.0, 31)
    L, B = np.meshgrid(lam, beta, indexing="ij")
    cp = cp_surface(L, B)
    ct = ct_surface(L, B)
    prov = ("REDUCED-ORDER analytical rotor surfaces (Heier/Slootweg C_P form + shaped "
            "C_T), affine-calibrated to published IEA-15MW anchors "
            f"(C_P,max={CP_MAX_ANCHOR} @ lambda={LAMBDA_OPT_ANCHOR}; C_T~{CT_REGION2_ANCHOR} "
            "region-2 -> ~2.4 MN rated thrust). NOT the NREL OpenFAST deck. See ASSUMPTIONS.md.")
    return AeroSurfaces(lam, beta, cp, ct, prov)


def _load_nrel_deck(path: str) -> AeroSurfaces:
    """Parse the OpenFAST-style Cp_Ct_Cq.txt rotor-performance table.

    The file lists a pitch-angle vector, a TSR vector, and blocks of C_P then C_T
    then C_q values on the (pitch x TSR) grid. We read C_P and C_T blocks.
    """
    with open(path, "r") as fh:
        raw = fh.read()
    tokens = raw.split("\n")

    def _find_vector(after_key: str) -> np.ndarray:
        for k, line in enumerate(tokens):
            if after_key.lower() in line.lower():
                # numbers are on the following non-empty line(s)
                for m in range(k + 1, len(tokens)):
                    nums = _floats(tokens[m])
                    if nums:
                        return np.array(nums)
        raise ValueError(f"Could not locate '{after_key}' in {path}")

    def _find_matrix(after_key: str, n_rows: int) -> np.ndarray:
        for k, line in enumerate(tokens):
            if after_key.lower() in line.lower():
                rows, m = [], k + 1
                while len(rows) < n_rows and m < len(tokens):
                    nums = _floats(tokens[m])
                    if nums:
                        rows.append(nums)
                    m += 1
                return np.array(rows)
        raise ValueError(f"Could not locate '{after_key}' in {path}")

    pitch = _find_vector("Pitch angle")
    tsr = _find_vector("TSR")
    # The Cp/Ct blocks are (TSR rows x pitch columns) = (lambda x beta) directly.
    cp_mat = _find_matrix("Power coefficient", len(tsr))
    ct_mat = _find_matrix("Thrust coefficient", len(tsr))
    return AeroSurfaces(
        lambda_grid=np.asarray(tsr, float),
        beta_grid=np.asarray(pitch, float),
        cp_table=np.asarray(cp_mat, float),      # [n_tsr, n_pitch] = [lambda, beta]
        ct_table=np.asarray(ct_mat, float),
        provenance=("IEA-15-240-RWT OpenFAST rotor-performance deck (Cp_Ct_Cq.txt), "
                    "loaded verbatim from models/data/. HIGH-FIDELITY rotor map."),
    )


def _floats(line: str) -> list[float]:
    out = []
    for tok in line.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            return []   # a non-numeric token means this isn't a pure data line
    return out
