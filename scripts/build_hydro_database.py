"""Offline BEM: compute the VolturnUS-S potential-flow hydrodynamic database (Capytaine).

Industry-standard hydrodynamics for a floating body: a boundary-element (potential-flow)
solve gives the frequency-dependent added mass A(omega), radiation damping B(omega), and
first-order wave-excitation force RAOs X(omega, heading) for the six rigid-body DOFs, plus
the hydrostatic stiffness. These feed a Cummins time-domain equation in ``physics/hydro.py``.

This runs OFFLINE (Capytaine is not a runtime dependency); the result is bundled as
``models/data/volturnus_hydro.npz`` and read by the live engine. Run:

    python scripts/build_hydro_database.py

Mesh: the VolturnUS-S semi — 1 central + 3 offset columns (D=12.5 m) and 3 pontoons
(12.5 x 7 m), submerged part only (draft 20 m). Reference point at the MSL origin (0,0,0);
the mass matrix (with COG offset and tower/RNA) is applied in the time-domain model.
"""
from __future__ import annotations

import os

import numpy as np

import capytaine as cpt
from capytaine import rigid_body_dofs
from capytaine.meshes.predefined import mesh_parallelepiped, mesh_vertical_cylinder

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "data")

# VolturnUS-S geometry (NREL/TP-5000-76773).
COL_R = 6.25
DRAFT = 20.0
SPACING = 51.75
PONTOON_W = 12.5
PONTOON_H = 7.0
RHO = 1025.0
G = 9.80665


def build_mesh():
    cols = []
    # Central column.
    cols.append(mesh_vertical_cylinder(length=DRAFT, radius=COL_R, center=(0, 0, -DRAFT / 2),
                                       resolution=(4, 24, 16)))
    # Three offset columns at 0/120/240 deg.
    for a in (0.0, 120.0, 240.0):
        cx, cy = SPACING * np.cos(np.radians(a)), SPACING * np.sin(np.radians(a))
        cols.append(mesh_vertical_cylinder(length=DRAFT, radius=COL_R,
                                           center=(cx, cy, -DRAFT / 2),
                                           resolution=(4, 24, 16)))
    # Three pontoons from centre to each offset column (bottom of the draft). Build each
    # along +x centred at the mid-span, then rotate about the origin to its heading.
    pons = []
    length = SPACING - 2 * COL_R
    for a in (0.0, 120.0, 240.0):
        pon = mesh_parallelepiped(size=(length, PONTOON_W, PONTOON_H),
                                  center=(SPACING / 2, 0.0, -DRAFT + PONTOON_H / 2),
                                  resolution=(12, 6, 4))
        pons.append(pon.rotated_z(np.radians(a)))
    mesh = cols[0]
    for m in cols[1:] + pons:
        mesh = mesh + m
    return mesh


def main():
    print("Building VolturnUS-S mesh…")
    mesh = build_mesh()
    body = cpt.FloatingBody(mesh=mesh, dofs=rigid_body_dofs(rotation_center=(0, 0, 0)),
                            center_of_mass=(0, 0, -2.0), name="VolturnUS-S")
    body = body.immersed_part()
    print(f"  panels (immersed): {body.mesh.nb_faces}")

    hs = body.compute_hydrostatic_stiffness()
    C_hydro = np.array(hs.values)                 # 6x6 hydrostatic stiffness

    omega = np.concatenate([np.linspace(0.05, 0.6, 20), np.linspace(0.65, 3.5, 40)])
    headings = np.radians([0.0, 30.0, 60.0, 90.0])
    print(f"Solving BEM: {len(omega)} frequencies x {len(headings)} headings…")
    solver = cpt.BEMSolver()
    dataset = solver.fill_dataset(_test_matrix(body, omega, headings), body)

    A = dataset["added_mass"].transpose("omega", "radiating_dof", "influenced_dof").values
    B = dataset["radiation_damping"].transpose("omega", "radiating_dof",
                                               "influenced_dof").values
    X = dataset["excitation_force"].transpose("omega", "wave_direction",
                                              "influenced_dof").values  # complex
    np.savez_compressed(
        os.path.join(DATA, "volturnus_hydro.npz"),
        omega=omega, headings=headings, added_mass=A, radiation_damping=B,
        excitation_force=X, hydrostatic_stiffness=C_hydro, rho=RHO, g=G,
        dof_order=np.array(list(dataset.radiating_dof.values), dtype=object))
    print("Saved models/data/volturnus_hydro.npz")
    # Quick sanity: infinite-frequency added mass and heave natural period estimate.
    print("A[surge,surge](low w):", round(float(A[0, 0, 0]) / 1e6, 1), "e6 kg")
    print("C_hydro heave (3,3):", round(C_hydro[2, 2] / 1e6, 2), "e6 N/m")


def _test_matrix(body, omega, headings):
    import xarray as xr
    return xr.Dataset(coords={
        "omega": omega, "wave_direction": headings,
        "radiating_dof": list(body.dofs), "water_depth": [200.0], "rho": [RHO]})


if __name__ == "__main__":
    main()
