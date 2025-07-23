"""Geometry utilities for drone racing control.

This module provides functions for vector normalization, tunnel geometry calculations,
obstacle avoidance, and path interpolation using cubic splines.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def unit(vector: np.ndarray) -> np.ndarray:
    """Returns the unit vector of the input vector, or [1.0, 0.0, 0.0] if the norm is too small.

    Args:
        vector: A numpy array representing the vector to normalize.

    Returns:
        A numpy array representing the normalized unit vector.
    """
    n = np.linalg.norm(vector)
    if n > 1e-6:
        return vector / n
    else:
        return np.array([1.0, 0.0, 0.0])

def tunnel_bounds(
    c: np.ndarray,
    c_next: np.ndarray,
    w_nom: float = 0.4,
    h_nom: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute center, normal, and binormal vectors and return them along with the nominal half-width/half-height of the tunnel.

    Args:
        c      : 3D path point at arc-length s
        c_next : 3D path point at arc-length s + Δs
        w_nom  : nominal tunnel half-width (meters)
        h_nom  : nominal tunnel half-height (meters)

    Returns:
        c      : unchanged center point (for consistency)
        n      : unit normal vector (lateral direction)
        b      : unit binormal vector (vertical-plane ‘twist’)
        w_nom  : nominal half-width
        h_nom  : nominal half-height
    """ 
    # Tangent vector
    tangent = c_next - c
    tangent /= (np.linalg.norm(tangent) + 1e-9)

    # Normal vector in the horizontal (XY) plane
    normal = np.array([-tangent[1], tangent[0], 0.0])
    normal /= (np.linalg.norm(normal) + 1e-9)

    # Binormal vector, perpendicular to both tangent and normal
    binormal = np.cross(tangent, normal)
    binormal /= (np.linalg.norm(binormal) + 1e-9)

    return c, normal, binormal, w_nom, h_nom

def move_tunnel_center(
    pref: np.ndarray,
    n_vec: np.ndarray,
    w_nom: float,
    obstacles: np.ndarray,
    obstacle_radius: list[float],
    look_ahead: float = 0.4,
    look_behind: float = 0.02
) -> np.ndarray:
    """Laterally shift the tunnel center so that every obstacle remains clear of the tunnel.
    
    Args:
        pref            : 3D tunnel center point [x, y, z]
        n_vec           : 3D unit-like normal vector defining the tunnel’s lateral axis
        w_nom           : nominal full tunnel width (meters)
        obstacles       : array of obstacle centers (shape (K, 3)); Z is ignored
        obstacle_radius : list of radii for each obstacle (meters)
        look_ahead      : forward window (meters) along the tunnel tangent to consider obstacles
        look_behind     : backward window (meters) for hysteresis

    Returns:
        new_center : 3D point (np.ndarray) of the shifted tunnel center
    """
    # Compute half-width and project pref into XY
    w_half = w_nom / 2.0
    pref_xy = pref[:2]

    # Normalize the lateral direction in XY
    n_xy = n_vec[:2]
    norm_n = np.linalg.norm(n_xy)
    if norm_n < 1e-9:
        # Degenerate normal: no shift
        return pref.copy()
    n_xy /= norm_n

    # Build the forward (tangential) direction in XY
    t_xy = np.array([ n_xy[1], -n_xy[0] ])

    # Accumulate required shift to clear each obstacle
    shift = 0.0
    default_radius = 0.10
    for i, obs_pt in enumerate(obstacles):
        if np.allclose(obs_pt, 0.0):
            continue  # skip placeholder entries

        p_xy = obs_pt[:2]
        r_obs = obstacle_radius[i] if i < len(obstacle_radius) else default_radius

        # Longitudinal test: skip if too far ahead or already behind
        d_long = float(t_xy @ (p_xy - pref_xy))
        if d_long >  look_ahead + r_obs or d_long < -look_behind - r_obs:
            continue

        # Lateral test: compute effective edge-to-edge distance
        d_lat = float(n_xy @ (p_xy - pref_xy))
        eff_dist = abs(d_lat) - r_obs
        if eff_dist < w_half:
            # obstacle intrudes: compute how much to shift so it just clears
            required = w_half - eff_dist
            shift += -np.sign(d_lat) * required

    # Apply lateral shift and reattach original Z
    new_xy = pref_xy + shift * n_xy
    new_center = np.array([ new_xy[0], new_xy[1], pref[2] ])
    return new_center


def pos_on_path(cs_x: CubicSpline, cs_y: CubicSpline, cs_z: CubicSpline, s_total: float, s: float) -> np.ndarray:
    """Get position on path at arc length s.
    
    Args:
        cs_x: CubicSpline for x-coordinate
        cs_y: CubicSpline for y-coordinate  
        cs_z: CubicSpline for z-coordinate
        s_total: Total arc length of the path
        s: Arc length parameter
        
    Returns:
        3D position at arc length s
    """
    s_wrap = s % s_total
    return np.array([cs_x(s_wrap), cs_y(s_wrap), cs_z(s_wrap)])



