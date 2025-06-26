"""Utility module."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from typing import TYPE_CHECKING, Type

import mujoco
import numpy as np
import toml
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

    from lsy_drone_racing.envs.race_core import RaceCoreEnv


logger = logging.getLogger(__name__)


def load_controller(path: Path) -> Type[Controller]:
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)

    def filter(mod: Any) -> bool:
        """Filter function to identify valid controller classes.

        Args:
            mod: Any attribute of the controller module to be checked.
        """
        subcls = inspect.isclass(mod) and issubclass(mod, Controller)
        return subcls and mod.__module__ == controller_module.__name__

    controllers = inspect.getmembers(controller_module, filter)
    controllers = [c for _, c in controllers if issubclass(c, Controller)]
    assert len(controllers) > 0, f"No controller found in {path}. Have you subclassed Controller?"
    assert len(controllers) == 1, f"Multiple controllers found in {path}. Only one is allowed."
    controller_module.Controller = controllers[0]
    assert issubclass(controller_module.Controller, Controller)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e


def load_config(path: Path) -> ConfigDict:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The configuration.
    """
    assert path.exists(), f"Configuration file not found: {path}"
    assert path.suffix == ".toml", f"Configuration file has to be a TOML file: {path}"

    with open(path, "r") as f:
        return ConfigDict(toml.load(f))


def draw_point(
    env: RaceCoreEnv,
    point: NDArray,  # shape (3,)
    size: float = 0.05,
    rgba: NDArray | None = None,
):
    """Draw a spherical marker at a given 3D point.

    Args:
        env: The drone racing environment.
        point: np.array([x, y, z]) position of the point.
        size: Radius of the sphere marker (in meters).
        rgba: Optional RGBA color (default: opaque green).
    """
    assert point.shape == (3,), "Point must be a 3D coordinate"

    sim = env.unwrapped.sim
    if sim.viewer is None:
        return
    viewer = sim.viewer.viewer

    if rgba is None:
        rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)  # opaque green

    size_vec = np.array([size, size, size], dtype=np.float32)
    # identity rotation for a sphere
    mat = np.eye(3, dtype=np.float32).reshape(-1)
    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=size_vec, pos=point, mat=mat, rgba=rgba
    )

# Draw a vertical cylinder obstacle given its top-center point
def draw_cylinder_obstacle(
    env: RaceCoreEnv,
    top_point: NDArray,  # shape (3,)
    radius: float = 0.1,
    height: float | None = None,
    rgba: NDArray | None = None,
):
    """Draw a vertical cylindrical obstacle given its top-center point.

    Args:
        env: The drone racing environment.
        top_point: np.array([x, y, z]) coordinates of the cylinder’s top center.
        radius: Radius of the cylinder (in meters).
        height: Height of the cylinder (in meters).
        rgba: Optional RGBA color (default: semi-transparent red).
    """
    top_point = np.asarray(top_point, dtype=np.float32)
    assert top_point.shape == (3,), "Top point must be a 3D coordinate"
    # if height not given, draw down to the ground (z=0)
    if height is None:
        height = top_point[2]

    sim = env.unwrapped.sim
    if sim.viewer is None:
        return
    viewer = sim.viewer.viewer

    # Default color: semi-transparent red
    if rgba is None:
        rgba = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)

    # Compute the cylinder's center: offset down by half the height
    center = top_point - np.array([0.0, 0.0, height / 2], dtype=np.float32)
    # Cylinder size vector: (radius, radius, half-height)
    size = np.array([radius, radius, height], dtype=np.float32)

    # Identity rotation for vertical alignment
    mat = np.eye(3, dtype=np.float32).reshape(-1)
    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=size,
        pos=center,
        mat=mat,
        rgba=rgba
    )


def draw_line(
    env: RaceCoreEnv,
    points: NDArray,
    rgba: NDArray | None = None,
    min_size: float = 3.0,
    max_size: float = 3.0,
):
    """Draw a line into the simulation.

    Args:
        env: The drone racing environment.
        points: An array of [N, 3] points that make up the line.
        rgba: The color of the line.
        min_size: The minimum line size. We linearly interpolate the size from min_size to max_size.
        max_size: The maximum line size.

    Note:
        This function has to be called every time before the env.render() step.
    """
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    sim = env.unwrapped.sim
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if sim.max_visual_geom < points.shape[0]:
        raise RuntimeError("Attempted to draw too many lines. Try to increase Sim.max_visual_geom")
    viewer = sim.viewer.viewer
    sizes = np.zeros_like(points)[:-1, :]
    sizes[:, 2] = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    sizes[:, :2] = np.linspace(min_size, max_size, len(sizes))[..., None]
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    mats = _rotation_matrix_from_points(points[:-1], points[1:]).as_matrix().reshape(-1, 9)
    for i in range(len(points) - 1):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_LINE, size=sizes[i], pos=points[i], mat=mats[i], rgba=rgba
        )


def draw_box(
    env: RaceCoreEnv,
    corner_min: NDArray,  # shape (3,)
    corner_max: NDArray,  # shape (3,)
    rgba: NDArray | None = None,
):
    """
    Draw a filled transparent 3D box from two opposite corners.

    Args:
        env: The drone racing environment.
        corner_min: np.array([x_min, y_min, z_min])
        corner_max: np.array([x_max, y_max, z_max])
        rgba: Optional RGBA color (default: semi-transparent red).
    """
    assert corner_min.shape == (3,) and corner_max.shape == (3,), "Each corner must be shape (3,)"

    sim = env.unwrapped.sim
    if sim.viewer is None:
        return
    viewer = sim.viewer.viewer

    if rgba is None:
        rgba = np.array([1.0, 0.0, 0.0, 0.3])  # semi-transparent red

    # Compute center and half-size
    center = (corner_min + corner_max) / 2
    extents = (corner_max - corner_min) / 2  # half-size per axis

    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=extents,
        pos=center,
        mat=np.eye(3).reshape(-1),  # identity rotation (AABB)
        rgba=rgba,
    )

def _rotation_matrix_from_points(p1: NDArray, p2: NDArray) -> R:
    """Robust: liefert immer eine gültige Rotation, auch bei Null-Segmenten."""
    v      = p2 - p1
    norms  = np.linalg.norm(v, axis=-1, keepdims=True)

    small           = norms < 1e-9           # praktisch Länge 0
    norms_safe      = norms.copy()
    norms_safe[small] = 1.0                 # Division vermeiden

    z_axis          = v / norms_safe
    z_axis[small.squeeze()] = np.array([0.0, 0.0, 1.0])    # Dummy-Richtung

    # Hilfsvektor wählen, der garantiert nicht kollinear ist
    helper = np.tile(np.array([1.0, 0.0, 0.0]), (z_axis.shape[0], 1))
    collinear = np.abs((z_axis * helper).sum(axis=-1)) > 0.99
    helper[collinear] = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(helper, z_axis)
    x_norm = np.linalg.norm(x_axis, axis=-1, keepdims=True)
    x_axis = x_axis / np.clip(x_norm, 1e-9, None)

    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))



def _quat_to_mat(q: NDArray) -> NDArray:
    """
    Wandelt einen Quaternion (x, y, z, w) in eine 3 × 3-Rotationsmatrix um.
    """
    x, y, z, w = q
    # sicherheitshalber normalisieren
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


# def draw_gates(
#     env: "RaceCoreEnv",
#     gates_pos: NDArray,  # (N,3)
#     gates_quat: NDArray,  # (N,4)  (x,y,z,w)
#     half_extents: NDArray | None = None,  # Loch-Halb­achsen  (x/2 , y/2 , z/2)
#     frame_thickness: float = 0.05,  # Balken­breite in Metern
#     rgba_opening: NDArray | None = None,  # Farbe des Lochs
#     rgba_frame: NDArray | None = None,  # Farbe der Balken
# ) -> None:
#     """
#     Zeichnet Gate-Öffnung **und** rote Umrandungs­balken.
#     """
#     # ------------------- Defaults ------------------------------------------------
#     if half_extents is None:
#         half_extents = np.array([0.225, 0.05, 0.225], dtype=np.float32)  # 0.45×0.45 Loch
#     if rgba_opening is None:
#         rgba_opening = np.array([0.0, 0.4, 1.0, 0.35], dtype=np.float32)  # semi-transparent
#     if rgba_frame is None:
#         rgba_frame = np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32)  # deckend-rot

#     sim = env.unwrapped.sim
#     if sim.viewer is None:  # Headless
#         return
#     viewer = sim.viewer.viewer
    
#     # ------------- Geometrie-Parameter ------------------------------------------
#     w, d, h = half_extents * 2  # volle Öffnungs­breite/-tiefe/-höhe
#     t = frame_thickness  # Balken­stärke (voll)
#     d_half = half_extents[1]  # halbe Tiefe Y

#     # Halbe Kanten­längen der vier Balken
#     size_vert = np.array([t / 2, d_half, (h + 2 * t) / 2], dtype=np.float32)
#     size_horiz = np.array([(w + 2 * t) / 2, d_half, t / 2], dtype=np.float32)

#     # Lokale Offsets der Balken­zentren
#     offs_left = np.array([-(w / 2 + t / 2), 0.0, 0.0], dtype=np.float32)
#     offs_right = -offs_left
#     offs_bottom = np.array([0.0, 0.0, -(h / 2 + t / 2)], dtype=np.float32)
#     offs_top = -offs_bottom
#     offsets = [
#         (offs_left, size_vert),
#         (offs_right, size_vert),
#         (offs_bottom, size_horiz),
#         (offs_top, size_horiz),
#     ]

#     # ------------- Rendering-Loop ----------------------------------------------
#     for pos, q in zip(gates_pos, gates_quat):
#         R = _quat_to_mat(q)

#         # 1) Öffnung
#         viewer.add_marker(
#             type=mujoco.mjtGeom.mjGEOM_BOX,
#             size=half_extents,weeks
#             pos=pos,
#             mat=R.reshape(-1),
#             rgba=rgba_opening,
#         )

#         # 2) Vier Rahmen­balken
#         for off_local, size in offsets:
#             off_world = R @ off_local
#             viewer.add_marker(
#                 type=mujoco.mjtGeom.mjGEOM_BOX,
#                 size=size,
#                 pos=pos + off_world,
#                 mat=R.reshape(-1),
#                 rgba=rgba_frame,
#             )

def draw_tunnel(env, pos_on_path, s_total,
                w=1.0, h=1.0, step=0.2,
                rgba=np.array([0.0, 0.8, 1.0, 0.12])):

    def local_frame(p, p_next):
        """Gib Normal-, Binormal- **und Tangente** zurück!"""
        t = p_next - p
        t /= np.linalg.norm(t) + 1e-9        # Tangente
        n = np.array([-t[1], t[0], 0.0])
        n /= np.linalg.norm(n) + 1e-9
        b = np.cross(t, n)
        b /= np.linalg.norm(b) + 1e-9
        return t, n, b                       # <-- t dazu

    s_vals = np.arange(0.0, s_total, step)
    for s in s_vals:
        p      = pos_on_path(s)
        p_next = pos_on_path(s + 0.01)
        t, n, b = local_frame(p, p_next)     # <- drei Rückgaben

        half_w, half_h, depth = w/2, h/2, step/2
        corner_min = p - half_w*n - half_h*b - depth*t
        corner_max = p + half_w*n + half_h*b + depth*t
        draw_box(env, corner_min, corner_max, rgba=rgba)


def draw_tube(env, pos_on_path, s_total,
              d_inner=0.50,
              thickness=0.02,
              step=0.50,         # 0.50 m!
              n_seg=8,           # 8 Segmente
              rgba=np.array([0.0, 0.6, 1.0, 0.13])):
    """
    Zeichnet eine halbtransparente Röhre um die Centerline.
    """
    if env.unwrapped.sim.viewer is None:
        return
    viewer = env.unwrapped.sim.viewer.viewer
    r = d_inner/2
    n_seg = 16                         # 22.5° Auflösung

    # vorberechnen
    angles = np.linspace(0, 2*np.pi, n_seg+1)[:-1]
    circle_xy = np.stack((np.cos(angles), np.sin(angles)), axis=-1) * r

    # Tangente & lokale Achsen wie gehabt
    def local_frame(p, p_next):
        t = (p_next - p); t /= np.linalg.norm(t)+1e-9
        n = np.array([-t[1], t[0], 0.0]); n/=np.linalg.norm(n)+1e-9
        b = np.cross(t,n); b/=np.linalg.norm(b)+1e-9
        return t,n,b

    for s in np.arange(0, s_total, step):
        p     = pos_on_path(s)
        p_next= pos_on_path(s+0.01)
        _, n, b = local_frame(p, p_next)

        # 16 kleine Quader-Stücke als Linien anlegen
        for i in range(n_seg):
            p1_local = circle_xy[i]
            p2_local = circle_xy[(i+1)%n_seg]

            world1 = p + p1_local[0]*n + p1_local[1]*b
            world2 = p + p2_local[0]*n + p2_local[1]*b
            seg_vec = world2 - world1
            length = np.linalg.norm(seg_vec)
            if length < 1e-4: continue

            size = np.array([thickness/2, thickness/2, length/2], dtype=np.float32)
            mat  = _rotation_matrix_from_points(world1, world2).as_matrix().reshape(-1)
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,
                pos=(world1+world2)/2,
                mat=mat,
                rgba=rgba,
            )

def draw_tube_splines(
    env,
    pos_on_path,
    s_total,
    d_inner=0.50,
    n_lines=12,
    step=0.2,
    rgba=np.array([0.0, 0.6, 1.0, 0.3]),
    thickness=2.0
):
    # Kreis-Offsets in lokalen (n,b) Koordinaten
    r = d_inner / 2
    angles = np.linspace(0, 2*np.pi, n_lines, endpoint=False)
    circle_xy = np.stack((np.cos(angles), np.sin(angles)), axis=-1) * r

    # Präpariere leere Listen für jede Linie
    line_pts = [ [] for _ in range(n_lines) ]

    # Hilfsfunktion wie in draw_tunnel
    def local_frame(p, p_next):
        t = p_next - p
        t /= np.linalg.norm(t) + 1e-9
        n = np.array([-t[1], t[0], 0.0])
        n /= np.linalg.norm(n) + 1e-9
        b = np.cross(t, n)
        b /= np.linalg.norm(b) + 1e-9
        return t, n, b

    # Alle s-Werte durchlaufen
    for s in np.arange(0, s_total, step):
        p      = pos_on_path(s)
        p_next = pos_on_path(min(s + 0.01, s_total))
        _, n, b = local_frame(p, p_next)

        # Für jede Linie den Welt-Punkt berechnen
        for i, (cx, cy) in enumerate(circle_xy):
            pt_world = p + cx * n + cy * b
            line_pts[i].append(pt_world)

    # Zeichne jede Linie in einem Rutsch
    for pts in line_pts:
        pts_arr = np.vstack(pts)  # Form (M,3)
        draw_line(
            env,
            pts_arr,
            rgba=rgba,
            min_size=thickness,
            max_size=thickness
        )

def draw_tube_dynamic(env, tube_cache,
                      n_circle=12, rgba=np.array([0.0,0.6,1.0,0.3]),
                      thickness=2.0):
    """Zeichnet Kreisbögen durch alle Cached-Segmente."""
    if not tube_cache:      # noch nichts gesammelt
        return
    # Kreis-Offsets vorbereiten
    angles = np.linspace(0, 2*np.pi, n_circle, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    # Für jede der n_circle-Linien Punkte sammeln
    lines = [[] for _ in range(n_circle)]
    for c,n,b,w,h in tube_cache:
        offs = circle * np.array([w/2, h/2])   # (12,2)
        for k,(ox,oy) in enumerate(offs):
            pt = c + ox*n + oy*b
            lines[k].append(pt)

    # Zeichnen
    for pts in lines:
        if len(pts) < 2:     # zu kurz
            continue
        draw_line(env, np.vstack(pts), rgba=rgba,
                  min_size=thickness, max_size=thickness)


# Draw tunnel cross-section outlines given corner points for each region
def draw_tunnel_regions_from_corners(
    env: "RaceCoreEnv",
    regions: np.ndarray,  # shape (N, 4, 3), corners for each stage
    rgba: np.ndarray | None = None,
    thickness: float = 2.0
) -> None:
    """
    Draws tunnel cross-section outlines given the 4 corner points for each stage.
    regions[j] should be a (4,3) array of corners in world coordinates for stage j.
    """
    sim = env.unwrapped.sim
    if sim.viewer is None:
        return
    viewer = sim.viewer.viewer

    N = regions.shape[0]
    assert regions.ndim == 3 and regions.shape[1:] == (4, 3), "regions must be shape (N,4,3)"

    if rgba is None:
        rgba = np.array([0.0, 1.0, 1.0, 0.3], dtype=np.float32)  # default cyan

    for j in range(N):
        poly = regions[j]
        # Close loop by appending first corner
        pts = np.vstack((poly, poly[0]))
        draw_line(env, pts, rgba=rgba, min_size=thickness, max_size=thickness)

# Draw gate constraint rectangles given the 4 corner points for each gate region
def draw_gate_constraints_from_corners(
    env: "RaceCoreEnv",
    regions: np.ndarray,  # shape (M, 4, 3), corners for each gate constraint region
    rgba: np.ndarray | None = None,
    thickness: float = 3.0
) -> None:
    """
    Draws gate constraint rectangles given the 4 corner points for each gate region.
    regions[j] should be a (4,3) array of corners in world coordinates for gate j.
    """
    sim = env.unwrapped.sim
    if sim.viewer is None:
        return
    viewer = sim.viewer.viewer

    M = regions.shape[0]
    assert regions.ndim == 3 and regions.shape[1:] == (4, 3), "regions must be shape (M,4,3)"

    if rgba is None:
        rgba = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)  # default semi-transparent red

    for j in range(M):
        poly = regions[j]
        # Close loop by appending first corner
        pts = np.vstack((poly, poly[0]))
        draw_line(env, pts, rgba=rgba, min_size=thickness, max_size=thickness)


def draw_obstacle_constraints(
    env: "RaceCoreEnv",
    obstacles_top_pos: np.ndarray,  # shape (4, 3): [x, y, z_top] jeder Säule
    radius: float = 0.2,
    rgba: np.ndarray | None = None
) -> None:
    """
    Zeichnet vertikale Zylinder‐Hindernisse durch Angabe ihrer Top‐Mittelpunkt‐Koordinaten.

    Args:
        env: Die DroneRacing‐Umgebung.
        obstacles_top_pos: ndarray der Form (4,3), jede Zeile ist [x, y, z_top] der Säulen‐Spitze.
                           Die Säule geht von Boden (z=0) bis z_top.
        radius: Radius jeder Zylinder‐Säule (Standard: 0.2 m).
        rgba: Optionaler RGBA‐Farbwert für die Zylinder (Standard: halbtransparentes Rot).
    """
    sim = env.unwrapped.sim
    if sim.viewer is None:
        return

    # Default‐Farbe: halbtransparentes Rot
    if rgba is None:
        rgba = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)

    # Jede Säule als Zylinder von z=0 bis z=z_top zeichnen
    for top_point in obstacles_top_pos:
        # top_point muss exakt 3 Komponenten haben: [x, y, z_top]
        assert top_point.shape == (3,), "Top-Point muss ein 3D-Koordinate sein"
        # draw_cylinder_obstacle erwartet (env, top_center, radius, height, rgba)
        # Höhe = top_point[2], Zylinder geht von 0 bis z_top
        draw_cylinder_obstacle(env, top_point, radius=radius, height=top_point[2], rgba=rgba)
def _rotation_matrix_from_points(p1: NDArray, p2: NDArray) -> R:
    """Generate rotation matrices that align their z-axis to p2-p1."""
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))
