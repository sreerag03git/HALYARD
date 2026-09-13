"""3-D mesh primitives (solid, lit) built from real geometry for the technical model.

Everything here is constructed geometry — solid cylinders, boxes, cones, lofted turbine
blades, and swept colored tubes — rendered as Plotly Mesh3d with lighting so the model reads
as a CAD assembly rather than a wireframe. No illustrative or imported art.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

_LIGHT = dict(ambient=0.45, diffuse=0.85, specular=0.25, roughness=0.55, fresnel=0.15)
_LIGHTPOS = dict(x=200, y=400, z=600)


def _mesh(verts, faces, color, opacity=1.0, name="", flat=True, show=False):
    v = np.asarray(verts, float)
    f = np.asarray(faces, int)
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                     color=color, opacity=opacity, flatshading=flat, name=name,
                     lighting=_LIGHT, lightposition=_LIGHTPOS, hoverinfo="skip",
                     showlegend=show)


def solid_cylinder(cx, cy, z0, z1, r, color, n=28, opacity=1.0, name="", show=False):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bot = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th), np.full(n, z0)])
    top = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th), np.full(n, z1)])
    cb = [[cx, cy, z0]]
    ct = [[cx, cy, z1]]
    verts = np.vstack([bot, top, cb, ct])
    ib, it, icb, ict = 0, n, 2 * n, 2 * n + 1
    faces = []
    for a in range(n):
        b = (a + 1) % n
        faces += [[ib + a, ib + b, it + b], [ib + a, it + b, it + a]]   # side
        faces += [[icb, ib + b, ib + a]]                                # bottom cap
        faces += [[ict, it + a, it + b]]                                # top cap
    return _mesh(verts, faces, color, opacity, name, show=show)


def box(cx, cy, cz, sx, sy, sz, color, yaw=0.0, opacity=1.0, name="", show=False):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    corners = np.array([[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    verts = corners @ R.T + np.array([cx, cy, cz])
    faces = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
             [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]]
    return _mesh(verts, faces, color, opacity, name, show=show)


def cone(apex, base_center, r, color, n=24, opacity=1.0, name="", show=False):
    apex = np.asarray(apex, float)
    bc = np.asarray(base_center, float)
    axis = bc - apex
    L = np.linalg.norm(axis) + 1e-9
    axis = axis / L
    tmp = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(axis, tmp); u /= np.linalg.norm(u) + 1e-9
    w = np.cross(axis, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = bc[None, :] + r * (np.cos(th)[:, None] * u[None, :] + np.sin(th)[:, None] * w[None, :])
    verts = np.vstack([apex[None, :], ring, bc[None, :]])
    faces = []
    for a in range(n):
        b = (a + 1) % n
        faces += [[0, 1 + a, 1 + b], [1 + n, 1 + b, 1 + a]]
    return _mesh(verts, faces, color, opacity, name, show=show)


def _airfoil(nc=10):
    """Thin cambered airfoil outline in (chord, thickness) unit coords, closed loop."""
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nc)))       # cosine spacing 0..1
    t = 0.10                                                # thickness/chord
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                  + 0.2843 * x ** 3 - 0.1015 * x ** 4)      # NACA-00xx half-thickness
    upper = np.column_stack([x, yt])
    lower = np.column_stack([x[::-1], -yt[::-1]])
    return np.vstack([upper, lower]) - np.array([0.3, 0.0])  # shift LE ref


def blade_mesh(hub, azimuth_deg, length, color, root_chord=5.4, tip_chord=0.6,
               precone_deg=4.0, root_r=3.0, n_span=14, name="", show=False):
    """Lofted, tapered, twisted 3-section-consistent turbine blade as a solid mesh.

    Rotor plane is y-z (rotor faces +x). Blade axis is radial in that plane at ``azimuth``;
    a small precone tilts it upwind (+x). Airfoil sections are scaled by a tapering chord and
    twisted from root to tip.
    """
    af = _airfoil()                                   # [P,2] (chord, thickness), unit chord
    P = len(af)
    hub = np.asarray(hub, float)
    a = np.radians(azimuth_deg)
    radial = np.array([0.0, np.cos(a), np.sin(a)])    # blade span direction (in rotor plane)
    pc = np.radians(precone_deg)
    span_dir = np.array([np.sin(pc), np.cos(pc) * np.cos(a), np.cos(pc) * np.sin(a)])
    span_dir /= np.linalg.norm(span_dir)
    chord_dir0 = np.array([0.0, -np.sin(a), np.cos(a)])   # tangential (chordwise at zero twist)
    thick_dir0 = np.array([1.0, 0.0, 0.0])                # ~ axial (thickness)
    rr = np.linspace(root_r, length, n_span)
    frac = (rr - root_r) / (length - root_r + 1e-9)
    chord = root_chord + (tip_chord - root_chord) * frac
    twist = np.radians(14.0 * (1 - frac) ** 1.4)          # root twist -> ~0 at tip
    verts = []
    for k in range(n_span):
        cdir = np.cos(twist[k]) * chord_dir0 + np.sin(twist[k]) * thick_dir0
        tdir = -np.sin(twist[k]) * chord_dir0 + np.cos(twist[k]) * thick_dir0
        center = hub + span_dir * rr[k]
        for p in range(P):
            pt = center + chord[k] * (af[p, 0] * cdir + af[p, 1] * tdir)
            verts.append(pt)
    verts = np.array(verts)
    faces = []
    for k in range(n_span - 1):
        for p in range(P):
            p1 = (p + 1) % P
            i0 = k * P + p; i1 = k * P + p1; i2 = (k + 1) * P + p; i3 = (k + 1) * P + p1
            faces += [[i0, i1, i3], [i0, i3, i2]]
    # tip cap (fan)
    tipc = len(verts)
    verts = np.vstack([verts, verts[(n_span - 1) * P:(n_span - 1) * P + P].mean(axis=0)])
    for p in range(P):
        p1 = (p + 1) % P
        faces += [[tipc, (n_span - 1) * P + p1, (n_span - 1) * P + p]]
    return _mesh(verts, faces, color, 1.0, name, show=show)


def tube(points, radius, color=None, intensity=None, colorscale="Turbo", n=10,
         colorbar_title="", name="", opacity=1.0, show=False):
    """Swept circular tube along a 3-D polyline; optionally colored by a per-node intensity."""
    pts = np.asarray(points, float)
    m = len(pts)
    tang = np.gradient(pts, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    ref = np.array([0.0, 1.0, 0.0])
    verts, inten = [], []
    rings = []
    for i in range(m):
        t = tang[i]
        u = np.cross(t, ref)
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(t, np.array([1.0, 0, 0]))
        u /= np.linalg.norm(u) + 1e-9
        w = np.cross(t, u)
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = pts[i][None, :] + radius * (np.cos(th)[:, None] * u[None, :]
                                           + np.sin(th)[:, None] * w[None, :])
        rings.append(ring)
        verts.append(ring)
        if intensity is not None:
            inten.extend([intensity[i]] * n)
    verts = np.vstack(verts)
    faces = []
    for i in range(m - 1):
        for a in range(n):
            b = (a + 1) % n
            i0 = i * n + a; i1 = i * n + b; i2 = (i + 1) * n + a; i3 = (i + 1) * n + b
            faces += [[i0, i1, i3], [i0, i3, i2]]
    f = np.asarray(faces, int)
    kw = dict(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
              flatshading=False, lighting=_LIGHT, lightposition=_LIGHTPOS, name=name,
              hoverinfo="skip", showlegend=show, opacity=opacity)
    if intensity is not None:
        kw.update(intensity=np.array(inten), colorscale=colorscale,
                  colorbar=dict(title=colorbar_title, len=0.5, x=0.0, thickness=14))
    else:
        kw.update(color=color)
    return go.Mesh3d(**kw)
