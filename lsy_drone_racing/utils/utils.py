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
    size: float = 0.01,
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


def draw_tunnel_bounds(
    env: "RaceCoreEnv",
    regions: np.ndarray,  # shape (N, 4, 3), corners for each stage
    rgba: np.ndarray | None = None,
    thickness: float = 3.0
) -> None:
    """Draws tunnel cross-section outlines given the 4 corner points for each stage.
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
