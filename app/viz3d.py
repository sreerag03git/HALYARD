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


def system_figure(shape: StaticShape, cable: DynamicCable, platform: Platform,
                  stress_along: np.ndarray | None = None,
                  color_label: str = "curvature (1/m)") -> go.Figure:
    """Static 3-D system model with the cable coloured by curvature or stress."""
    depth = cable.water_depth_m
    extent_x = (min(-80, shape.x.min() - 20), max(shape.x.max() + 20, 120))
    extent_y = (-80, 80)
    fig = go.Figure()

    # Water surface and seabed.
    fig.add_trace(_plane(0.0, extent_x, extent_y, "#BFD3E6", 0.18))
    fig.add_trace(_plane(-depth, extent_x, extent_y, "#D8CBB0", 0.35))

    # Platform columns + pontoons.
    cols = _column_positions(platform)
    R = platform.column_diameter_m / 2
    for (cx, cy) in cols:
        fig.add_trace(_cylinder(cx, cy, -platform.draft_m, platform.freeboard_m, R,
                                color=ACCENT, opacity=0.55))
    cx0, cy0 = cols[-1]
    for (cx, cy) in cols[:-1]:
        fig.add_trace(go.Scatter3d(x=[cx0, cx], y=[cy0, cy],
                                   z=[-platform.draft_m + 3, -platform.draft_m + 3],
                                   mode="lines", line=dict(color=ACCENT, width=10),
                                   hoverinfo="skip", showlegend=False))

    # Tower + rotor disc.
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[platform.freeboard_m, platform.hub_height_m],
                               mode="lines", line=dict(color=GREY, width=8),
                               hoverinfo="skip", showlegend=False, name="tower"))
    disc_t = np.linspace(0, 2 * np.pi, 40)
    Rrot = 120.0
    fig.add_trace(go.Scatter3d(x=np.zeros_like(disc_t), y=Rrot * np.cos(disc_t),
                               z=platform.hub_height_m + Rrot * np.sin(disc_t),
                               mode="lines", line=dict(color=INK, width=3),
                               hoverinfo="skip", name="rotor", showlegend=True))

    # Mooring lines (catenary) from fairleads to anchors.
    for (cx, cy) in cols[:-1]:
        ux, uy = cx / np.hypot(cx, cy), cy / np.hypot(cx, cy)
        ax_, ay_ = cx + ux * 600, cy + uy * 600
        s = np.linspace(0, 1, 30)
        mx = cx + s * (ax_ - cx)
        my = cy + s * (ay_ - cy)
        mz = -platform.fairlead_depth_m + (-depth + platform.fairlead_depth_m) * (s ** 1.7)
        fig.add_trace(go.Scatter3d(x=mx, y=my, z=mz, mode="lines",
                                   line=dict(color=GREY, width=2), opacity=0.6,
                                   hoverinfo="skip", showlegend=False))

    # Dynamic lazy-wave cable, coloured along its length.
    color = stress_along if stress_along is not None else shape.curvature
    fig.add_trace(go.Scatter3d(
        x=shape.x, y=np.zeros_like(shape.x), z=shape.z, mode="lines+markers",
        line=dict(color=color, width=7, colorscale="Viridis",
                  colorbar=dict(title=color_label, len=0.5, x=0.02)),
        marker=dict(size=2, color=color, colorscale="Viridis", showscale=False),
        name="dynamic cable"))

    # Buoyancy modules (spheres) along the buoyancy section.
    L = cable.total_length_m
    b0, b1 = cable.buoyancy_start_frac * L, cable.buoyancy_end_frac * L
    bmask = (shape.s >= b0) & (shape.s <= b1)
    idx = np.where(bmask)[0][::3]
    fig.add_trace(go.Scatter3d(x=shape.x[idx], y=np.zeros_like(idx), z=shape.z[idx],
                               mode="markers", marker=dict(size=6, color=WARN, opacity=0.8),
                               name="buoyancy modules"))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID_LINE),
            yaxis=dict(title="y (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID_LINE),
            zaxis=dict(title="z (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID_LINE),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.4, z=0.7)),
        ),
        margin=dict(l=0, r=0, t=10, b=0), height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=GRID_LINE, borderwidth=1),
    )
    return fig


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
