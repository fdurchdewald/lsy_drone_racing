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
    """Generate rotation matrices that align their z-axis to p2-p1."""
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
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


def draw_gates(
    env: "RaceCoreEnv",
    gates_pos: NDArray,  # (N,3)
    gates_quat: NDArray,  # (N,4)  (x,y,z,w)
    half_extents: NDArray | None = None,  # Loch-Halb­achsen  (x/2 , y/2 , z/2)
    frame_thickness: float = 0.05,  # Balken­breite in Metern
    rgba_opening: NDArray | None = None,  # Farbe des Lochs
    rgba_frame: NDArray | None = None,  # Farbe der Balken
) -> None:
    """
    Zeichnet Gate-Öffnung **und** rote Umrandungs­balken.
    """
    # ------------------- Defaults ------------------------------------------------
    if half_extents is None:
        half_extents = np.array([0.225, 0.05, 0.225], dtype=np.float32)  # 0.45×0.45 Loch
    if rgba_opening is None:
        rgba_opening = np.array([0.0, 0.4, 1.0, 0.35], dtype=np.float32)  # semi-transparent
    if rgba_frame is None:
        rgba_frame = np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32)  # deckend-rot

    sim = env.unwrapped.sim
    if sim.viewer is None:  # Headless
        return
    viewer = sim.viewer.viewer

    # ------------- Geometrie-Parameter ------------------------------------------
    w, d, h = half_extents * 2  # volle Öffnungs­breite/-tiefe/-höhe
    t = frame_thickness  # Balken­stärke (voll)
    d_half = half_extents[1]  # halbe Tiefe Y

    # Halbe Kanten­längen der vier Balken
    size_vert = np.array([t / 2, d_half, (h + 2 * t) / 2], dtype=np.float32)
    size_horiz = np.array([(w + 2 * t) / 2, d_half, t / 2], dtype=np.float32)

    # Lokale Offsets der Balken­zentren
    offs_left = np.array([-(w / 2 + t / 2), 0.0, 0.0], dtype=np.float32)
    offs_right = -offs_left
    offs_bottom = np.array([0.0, 0.0, -(h / 2 + t / 2)], dtype=np.float32)
    offs_top = -offs_bottom
    offsets = [
        (offs_left, size_vert),
        (offs_right, size_vert),
        (offs_bottom, size_horiz),
        (offs_top, size_horiz),
    ]

    # ------------- Rendering-Loop ----------------------------------------------
    for pos, q in zip(gates_pos, gates_quat):
        R = _quat_to_mat(q)

        # 1) Öffnung
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_extents,
            pos=pos,
            mat=R.reshape(-1),
            rgba=rgba_opening,
        )

        # 2) Vier Rahmen­balken
        for off_local, size in offsets:
            off_world = R @ off_local
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,
                pos=pos + off_world,
                mat=R.reshape(-1),
                rgba=rgba_frame,
            )
