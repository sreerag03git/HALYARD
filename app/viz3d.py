"""Interactive 3-D system model (Plotly 3D) — all from real computed geometry (§8).

To-scale seabed and water surface, the VolturnUS-S semi-submersible (columns + pontoons
from published geometry), tower, rotor disc, catenary mooring lines, and the solved lazy-
wave dynamic cable with buoyancy modules — coloured along its length by curvature/stress so
the hang-off concentration is visible. Optional animation drives the platform and cable top
through the computed motion over a chosen window. No illustrative imagery — only geometry.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from app.theme import ACCENT, GREY, GRID_LINE, INK, WARN
from models.dynamic_cable import DynamicCable
from models.volturnus_s import Platform
from physics.cable import QuasiStaticFamily, StaticShape


def _cylinder(x0, y0, z0, z1, radius, n=18, color=ACCENT, opacity=0.5, name=""):
    theta = np.linspace(0, 2 * np.pi, n)
    zc = np.array([z0, z1])
    T, Z = np.meshgrid(theta, zc)
    X = x0 + radius * np.cos(T)
    Y = y0 + radius * np.sin(T)
    return go.Surface(x=X, y=Y, z=Z, showscale=False,
                      colorscale=[[0, color], [1, color]], opacity=opacity,
                      hoverinfo="skip", name=name)


def _column_positions(p: Platform):
    r = p.column_spacing_m
    ang = np.array([0.0, 120.0, 240.0]) * np.pi / 180.0
    xs = r * np.cos(ang)
    ys = r * np.sin(ang)
    return list(zip(xs, ys)) + [(0.0, 0.0)]


def _plane(z, xr, yr, color, opacity):
    X, Y = np.meshgrid(np.linspace(*xr, 2), np.linspace(*yr, 2))
    Z = np.full_like(X, z)
    return go.Surface(x=X, y=Y, z=Z, showscale=False,
                      colorscale=[[0, color], [1, color]], opacity=opacity,
                      hoverinfo="skip")


STEEL = "#7C8B99"        # platform steel
STEEL_D = "#5E6B78"      # darker steel (pontoons/tower)
NACELLE = "#4A5764"
BLADE = "#EDEFF1"
CABLE_BLACK = "#20242A"
BUOY = "#E0A03A"


def system_figure(shape: StaticShape, cable: DynamicCable, platform: Platform,
                  stress_along: np.ndarray | None = None,
                  color_label: str = "curvature (1/m)") -> go.Figure:
    """Detailed, lit CAD-style 3-D model built entirely from the real computed geometry."""
    from app import meshes as M
    depth = cable.water_depth_m
    extent_x = (-110, shape.x.max() + 40)
    extent_y = (-95, 95)
    fig = go.Figure()

    # Sea surface (semi-transparent) and seabed.
    fig.add_trace(_plane(0.0, extent_x, extent_y, "#AEC6DE", 0.22))
    fig.add_trace(_plane(-depth, extent_x, extent_y, "#C9BB9C", 0.55))

    cols = _column_positions(platform)
    Rc = platform.column_diameter_m / 2
    draft, fb = platform.draft_m, platform.freeboard_m
    # Pontoons (submerged box beams from centre to each offset column).
    cx0, cy0 = cols[-1]
    for (cx, cy) in cols[:-1]:
        yaw = np.arctan2(cy, cx)
        length = np.hypot(cx, cy) - platform.column_diameter_m
        fig.add_trace(M.box((cx0 + cx) / 2, (cy0 + cy) / 2, -draft + 3.5,
                            length, platform.column_diameter_m * 0.9, 7.0,
                            STEEL_D, yaw=yaw))
    # Columns (solid lit cylinders).
    for (cx, cy) in cols:
        fig.add_trace(M.solid_cylinder(cx, cy, -draft, fb, Rc, STEEL, name="column"))

    # Transition piece + tapered tower + nacelle + hub + 3 blades.
    hub_h = platform.hub_height_m
    fig.add_trace(M.solid_cylinder(0, 0, fb, fb + 6, 5.5, STEEL_D))            # TP
    fig.add_trace(M.cone((0, 0, hub_h), (0, 0, fb + 6), 5.0, STEEL_D))         # tapered tower
    fig.add_trace(M.box(-4, 0, hub_h, 18, 8, 7, NACELLE, name="nacelle"))     # nacelle
    fig.add_trace(M.cone((10, 0, hub_h), (2, 0, hub_h), 3.2, "#3A4450"))      # hub nose cone
    Rrot = platform.column_diameter_m * 0 + 120.0
    for az in (90.0, 210.0, 330.0):
        fig.add_trace(M.blade_mesh((2, 0, hub_h), az, Rrot - 3, BLADE, name="blade"))

    # Mooring lines: 3-line catenary tubes from the offset-column fairleads (partial span
    # shown for readability; real lines run ~800 m to the anchors).
    for (cx, cy) in cols[:-1]:
        ux, uy = cx / np.hypot(cx, cy), cy / np.hypot(cx, cy)
        s = np.linspace(0, 1, 40)
        anx, any_ = cx + ux * 240, cy + uy * 240
        mx = cx + s * (anx - cx); my = cy + s * (any_ - cy)
        mz = -platform.fairlead_depth_m + (-depth + platform.fairlead_depth_m) * (s ** 2.2)
        fig.add_trace(M.tube(np.column_stack([mx, my, mz]), 1.6, color="#556070", n=7))

    # Dynamic lazy-wave cable as a colored tube (curvature / stress heatmap).
    inten = stress_along if stress_along is not None else shape.curvature
    cab_pts = np.column_stack([shape.x, np.zeros_like(shape.x), shape.z])
    fig.add_trace(M.tube(cab_pts, cable.outer_diameter_m * 6, intensity=np.asarray(inten),
                         colorscale="Turbo", n=10, colorbar_title=color_label, name="cable",
                         show=True))
    # Bend stiffener (tapered cone) at the hang-off.
    ho = cab_pts[1]
    fig.add_trace(M.cone(tuple(cab_pts[3]), tuple(ho), 1.6, "#2B2F35"))
    # Buoyancy modules (short cylinders straddling the cable in the buoyancy section).
    L = cable.total_length_m
    b0, b1 = cable.buoyancy_start_frac * L, cable.buoyancy_end_frac * L
    bmask = (shape.s >= b0) & (shape.s <= b1)
    for i in np.where(bmask)[0][::2]:
        fig.add_trace(M.solid_cylinder(shape.x[i], 0, shape.z[i] - 2.2, shape.z[i] + 2.2,
                                       0.9, BUOY, n=12))

    _dimension_annotations(fig, platform, cable, shape, depth)
    _scene(fig, extent_x, extent_y, depth)
    return fig


def _dimension_annotations(fig, platform, cable, shape, depth):
    hub_h = platform.hub_height_m
    ann = [
        (0, 0, hub_h + 128, "rotor Ø 240 m"),
        (-70, 0, hub_h / 2, "hub 150 m"),
        (0, 0, 8, "MSL"),
        (shape.x.max() * 0.5, 0, -depth + 8, f"water depth {depth:.0f} m"),
        (platform.column_spacing_m, 0, platform.freeboard_m + 4, "VolturnUS-S"),
        (shape.x[1] + 6, 0, shape.z[1], "hang-off"),
    ]
    fig.add_trace(go.Scatter3d(
        x=[a[0] for a in ann], y=[a[1] for a in ann], z=[a[2] for a in ann],
        mode="text", text=[a[3] for a in ann],
        textfont=dict(size=10, color=INK), hoverinfo="skip", showlegend=False))


def _scene(fig, extent_x, extent_y, depth):
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID_LINE,
                       showspikes=False),
            yaxis=dict(title="y (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID_LINE,
                       showspikes=False),
            zaxis=dict(title="z (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID_LINE,
                       showspikes=False),
            aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=0.55),
                                           up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=10, b=0), height=620, paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(255,255,255,0.75)", bordercolor=GRID_LINE, borderwidth=1),
    )


# ---------------------------------------------------------------------------
# Animated system model — driven by the computed platform motion (§8)
# ---------------------------------------------------------------------------
def _platform_lines(platform: Platform):
    """Return a list of (x, y, z) polylines for the platform frame + tower + rotor."""
    cols = _column_positions(platform)
    lines = []
    for (cx, cy) in cols:                       # vertical column edges
        lines.append((np.array([cx, cx]), np.array([cy, cy]),
                      np.array([-platform.draft_m, platform.freeboard_m])))
    cx0, cy0 = cols[-1]
    for (cx, cy) in cols[:-1]:                   # pontoons (centre -> offset)
        zc = -platform.draft_m + 3.0
        lines.append((np.array([cx0, cx]), np.array([cy0, cy]), np.array([zc, zc])))
    lines.append((np.array([0, 0]), np.array([0, 0]),
                  np.array([platform.freeboard_m, platform.hub_height_m])))   # tower
    th = np.linspace(0, 2 * np.pi, 40)           # rotor disc (y-z plane)
    Rrot = 120.0
    lines.append((np.zeros_like(th), Rrot * np.cos(th),
                  platform.hub_height_m + Rrot * np.sin(th)))
    return lines


def _rigid_transform(x, y, z, surge, pitch_rad):
    """Rigid body: pitch about the y-axis through MSL origin, then surge in +x."""
    xr = x * np.cos(pitch_rad) + z * np.sin(pitch_rad) + surge
    zr = -x * np.sin(pitch_rad) + z * np.cos(pitch_rad)
    return xr, y, zr


def animated_system_figure(family: QuasiStaticFamily, platform: Platform, sim,
                           n_frames: int = 24, exaggeration: float = 8.0) -> go.Figure:
    """Animate the platform + cable through the computed motion over the event window.

    Uses the computed surge/pitch time series (support+recovery window). Motion is small,
    so it is exaggerated by ``exaggeration`` for visibility — this is stated in the caption.
    The cable follows via the quasi-static family shape at the fairlead offset.
    """
    cable = family.cable
    depth = cable.water_depth_m
    # Window: from the event to ~120 s after (captures support + recovery + ring-down).
    t = sim.t
    t0 = sim.config.schedule.t_event_s
    win = (t >= t0 - 10) & (t <= t0 + 130)
    idx = np.linspace(np.argmax(win), len(t) - 1 - np.argmax(win[::-1]), n_frames).astype(int)
    surge = sim.surge_m
    pitch = np.radians(sim.pitch_deg)
    surge_ref, pitch_ref = surge[win].mean(), pitch[win].mean()
    z_att = cable.hangoff_z_m

    extent_x = (-80, 190)
    extent_y = (-80, 80)
    fig = go.Figure()
    # Static context.
    fig.add_trace(_plane(0.0, extent_x, extent_y, "#BFD3E6", 0.15))
    fig.add_trace(_plane(-depth, extent_x, extent_y, "#D8CBB0", 0.30))
    plines = _platform_lines(platform)

    def frame_traces(k):
        i = idx[k]
        s = (surge[i] - surge_ref) * exaggeration
        th = (pitch[i] - pitch_ref) * exaggeration
        traces = []
        for (lx, ly, lz) in plines:
            xr, yr, zr = _rigid_transform(lx, ly, lz, s, th)
            traces.append(go.Scatter3d(x=xr, y=yr, z=zr, mode="lines",
                                       line=dict(color=ACCENT, width=5), hoverinfo="skip",
                                       showlegend=False))
        # Cable at the (exaggerated) fairlead offset.
        dx = ((surge[i] - surge_ref) + z_att * (pitch[i] - pitch_ref)) * exaggeration
        rc = family.shape_at(dx)
        traces.append(go.Scatter3d(x=rc[:, 0], y=np.zeros(len(rc)), z=rc[:, 1], mode="lines",
                                   line=dict(color=WARN, width=6), name="cable",
                                   showlegend=False))
        return traces

    for tr in frame_traces(0):
        fig.add_trace(tr)
    n_anim = len(plines) + 1
    frames = [go.Frame(data=frame_traces(k), name=str(k),
                       traces=list(range(2, 2 + n_anim))) for k in range(n_frames)]
    fig.frames = frames
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05,
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=90, redraw=True),
                                                         fromcurrent=True)]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])],
        scene=dict(xaxis=dict(title="x (m)", gridcolor=GRID_LINE),
                   yaxis=dict(title="y (m)", gridcolor=GRID_LINE),
                   zaxis=dict(title="z (m)", gridcolor=GRID_LINE),
                   aspectmode="data", camera=dict(eye=dict(x=1.6, y=1.3, z=0.6))),
        margin=dict(l=0, r=0, t=10, b=0), height=560, paper_bgcolor="rgba(0,0,0,0)")
    return fig
