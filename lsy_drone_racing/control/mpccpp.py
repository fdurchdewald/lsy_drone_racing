"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.geometry import move_tunnel_center, pos_on_path, tunnel_bounds, unit
from lsy_drone_racing.control.mpccpp_config import MPCC_CFG
from lsy_drone_racing.control.ocp_solver import create_ocp_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray

class MPController(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
        PARAM_DICT: dict[str, float] | None = None,
    ):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
            PARAM_DICT: Optional dictionary of parameter overrides for the MPC configuration.
        """
        super().__init__(obs, info, config)
        
        # The PARAM_DICT is passed from the environment to allow for dynamic configuration, only used for TuRBO tuning.
        if PARAM_DICT is not None:
            MPCC_CFG.update(PARAM_DICT)
       
        # initialize waypoints
        gates_pos = obs["gates_pos"] 
        waypoints = np.array(
                    [
                        obs["pos"],
                        [0.95, 1.0, 0.3],
                        gates_pos[0],  # gate0 centre
                        [obs["obstacles_pos"][1][0] - 0.3, obs["obstacles_pos"][1][1], 0.75],
                        [obs["obstacles_pos"][1][0] - 0.3, obs["obstacles_pos"][1][1] - 0.4, 0.85],
                        gates_pos[1],  # gate1 centre
                        [0.5, 0.1, 0.8],
                        gates_pos[2],  # gate2 centre
                        [gates_pos[2][0] - 0.21, gates_pos[2][1] + 0.21, gates_pos[2][2] + 0.18],
                        [obs["obstacles_pos"][3][0] + 0.1, obs["obstacles_pos"][3][1], 1.1],
                        gates_pos[3],  # gate3 centre
                        [-0.5, -2, 1.11],
                        [-0.5, -6, 1.11],
                    ]
                )
        self._waypoints = waypoints
        self._prev_waypoints = self._waypoints.copy()
        self._gate_to_wp_index = {0: 2, 1: 5, 2: 7, 3: 10}
        
        # create cubic spline for the trajectory
        seg_lens = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        s_grid = np.hstack(([0.0], np.cumsum(seg_lens)))  
        self.s_total = float(s_grid[-1])

        cs_x = CubicSpline(s_grid, waypoints[:, 0])
        cs_y = CubicSpline(s_grid, waypoints[:, 1])
        cs_z = CubicSpline(s_grid, waypoints[:, 2])
        self._cs_x, self._cs_y, self._cs_z = cs_x, cs_y, cs_z
        vis_s = np.linspace(0.0, self.s_total, 200, endpoint=False)
        self.vis_s = vis_s
        self.traj_points = np.column_stack((cs_x(vis_s), cs_y(vis_s), cs_z(vis_s)))

        # OCP solver settings
        self._tick = 0
        self.N = MPCC_CFG["N"] 
        self.T_HORIZON = MPCC_CFG["T_HORIZON"]  
        self.dt = self.T_HORIZON / self.N
        self._acados_fail = 0
        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)
        self.last_f_collective = 0.2
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.2
        self.last_theta = 0.3
        self.last_vtheta = 0.1
        self.finished = False
        self._predicted_theta = np.zeros(self.N + 1)  

        # error handling
        self._bw_ramp_elapsed = 0.0
        self._prev_idx_min_vis = 0

        # save initial obstacles positions for comparison later
        self._init_obs = np.asarray(obs["obstacles_pos"])

        # save planned trajectory and drone trajectory for visualization
        self._planned_traj = np.zeros((self.N + 1, 3))
        self._drone_traj = []
        self._prev_tunnel_centers = None

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        # Reset tunnel for each call
        self._tunnel_wh = []
        self._tunnel_nb  = []
        self._tunnel_centers = []
        prev_centers = self._prev_tunnel_centers

        # update trajectory based on the current observation
        self.update_trajectory(obs)

        # bw ramping if solver failed
        if self._bw_ramp_elapsed is not None:
            self._bw_ramp_elapsed += self.dt

        # extract current position and orientation
        pos = obs["pos"]
        quat = obs["quat"]
        r = R.from_quat(quat)
        rpy = r.as_euler("xyz", degrees=False) 

        # project current position onto path to get up‑to‑date arclength s_cur
        traj_points_vis = self.traj_points  
        dists_vis = np.linalg.norm(traj_points_vis - pos, axis=1)
        idx_min_vis = int(np.argmin(dists_vis))
        if abs(self._prev_idx_min_vis - idx_min_vis) < 10:
            s_cur = self.vis_s[idx_min_vis]
            self._prev_idx_min_vis = idx_min_vis
        else:
            s_cur = self.vis_s[self._prev_idx_min_vis]
            print("Warning: idx_min_vis jump detected, using previous s_cur value:")

        # set current state for ocp solver
        xcurrent = np.zeros(16)
        xcurrent[:3] = obs["pos"]
        xcurrent[3:6] = obs["vel"]
        xcurrent[6:9] = rpy 
        xcurrent[9] = self.last_f_collective
        xcurrent[10] = self.last_f_cmd
        xcurrent[11:14] = self.last_rpy_cmd
        xcurrent[14] = self.last_theta
        xcurrent[15] = self.last_vtheta

        # predict theta trajectory
        try:
            vtheta_pred = np.zeros(self.N)
            for j in range(1, self.N + 1):
                xj = self.acados_ocp_solver.get(j, "x")
                vtheta_pred[j - 1] = float(xj[15])
            theta_pred = np.zeros(self.N + 1)
            theta_pred[0] = s_cur
            for j in range(self.N):
                theta_pred[j + 1] = theta_pred[j] + vtheta_pred[j] * self.dt
        except Exception:
            theta_pred = self.last_theta + self.last_vtheta * np.linspace(
                0.0, self.T_HORIZON, self.N + 1
            )
            theta_pred[0] = s_cur
        self._predicted_theta = theta_pred.copy()

        # define start up for the OCP solver and tunnel bounds
        if self._tick == 0:
            s_cur = 0.0
            self._predicted_theta = np.zeros(self.N + 1)
            self.acados_ocp_solver.options_set("qp_warm_start", 0) 
        else:
            self.acados_ocp_solver.options_set("qp_warm_start", 2)  

        # set the initial state for the OCP solver
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        # loop over the prediction horizon
        for j in range(self.N):
            # set reference point and tangent vector
            s_ref = theta_pred[j]
            pref = pos_on_path(self._cs_x, self._cs_y, self._cs_z, self.s_total, s_ref)
            pref_next = pos_on_path(self._cs_x, self._cs_y, self._cs_z, self.s_total, s_ref + 0.001)
            tangent = unit(pref_next - pref)

            # extract tunnel dimensions
            w_far = MPCC_CFG["TUNNEL_WIDTH"]
            h_far = MPCC_CFG["TUNNEL_WIDTH"]
            w_gate = MPCC_CFG["TUNNEL_WIDTH_GATE"]
            h_gate = MPCC_CFG["TUNNEL_WIDTH_GATE"]
            flat_dist = MPCC_CFG["GATE_FLAT_DIST"]
            narrow_dist = MPCC_CFG["NARROW_DIST"]
            
            # define the tunnel narrowing mechanism based on the distance to the gate
            gate_pos = obs["gates_pos"][int(obs["target_gate"])]
            dist_gate = np.linalg.norm(pref - gate_pos)
            for i, gate_pos in enumerate(obs["gates_pos"]):
                dist_gate = np.linalg.norm(pref - gate_pos)
                if dist_gate > narrow_dist:
                    # far from gate: full tunnel width
                    w_tunnel = w_far
                    h_tunnel = h_far
                elif dist_gate > flat_dist:
                    # narrowing region: interpolate between wide and gate widths
                    ratio = (dist_gate - flat_dist) / (narrow_dist - flat_dist)
                    w_tunnel = ratio * w_far + (1.0 - ratio) * w_gate[i]
                    h_tunnel = ratio * h_far + (1.0 - ratio) * h_gate[i]
                    break
                else:
                    # within flat region near the gate: constant gate width
                    w_tunnel = w_gate[i]
                    h_tunnel = h_gate[i]
                    break

            # Compute tunnel orientation with the stage‑specific width/height
            c      = pos_on_path(self._cs_x, self._cs_y, self._cs_z, self.s_total, s_ref)
            c_next = pos_on_path(self._cs_x, self._cs_y, self._cs_z, self.s_total, s_ref + 0.001)
            c, n, b, w, h = tunnel_bounds(
                c, c_next, w_tunnel, h_tunnel,
            )
            self._tunnel_nb.append((n, b))

            # Shift Tunnel Centre based on observations only
            current_obs = np.asarray(obs["obstacles_pos"])
            init_obs = self._init_obs
            obs_arr = np.zeros_like(init_obs)
            mask = np.array([not np.allclose(cur, init) for cur, init in zip(current_obs, init_obs)])
            obs_arr[mask] = current_obs[mask]
            pref_shift = move_tunnel_center(pref, n, w_tunnel, obs_arr, obstacle_radius=MPCC_CFG["OBSTACLE_RADIUS"],)

            # Smooth the tunnel centre with the previous stage's centre
            if prev_centers is not None and len(prev_centers) > j:
                prev_c = prev_centers[j]
            else:
                prev_c = pref
            smoothed_c = prev_c + MPCC_CFG["ALPHA_INTERP"] * (pref_shift - prev_c)
            self._tunnel_wh.append((w, h))
            self._tunnel_centers.append(smoothed_c)

            # Parameter ramp-up at start
            ramp_start = min(1.0, self._tick * self.dt / MPCC_CFG["RAMP_TIME"])
            qc_val = MPCC_CFG["QC"] * (0.5 + 0.5 * ramp_start)
            ql_val = MPCC_CFG["QL"] * (0.5 + 0.5 * ramp_start)
            mu_val = MPCC_CFG["MU"] * ramp_start
            if self._bw_ramp_elapsed is not None:
                ramp = min(1.0, self._bw_ramp_elapsed / MPCC_CFG["BW_RAMP"])
                bw_val = MPCC_CFG["BARRIER_WEIGHT"] * ramp**2
            else:
                bw_val = MPCC_CFG["BARRIER_WEIGHT"]
            
            # Set the parameters for the current stage in the OCP solver
            p_vec = np.concatenate(
                [
                    smoothed_c,
                    tangent,
                    [qc_val, ql_val, mu_val],
                    n,
                    b,
                    [w / 2, h / 2, bw_val],
                ]
            )
            self.acados_ocp_solver.set(j, "p", p_vec)
        
        # fill in previous tunnel centers
        self._prev_tunnel_centers = list(self._tunnel_centers)

        # Measure time taken for solver
        tic = time.perf_counter()

        # Definition on how to react to solver failures
        status = self.acados_ocp_solver.solve()
        toc = time.perf_counter()
        self._last_solve_ms = (toc - tic) * 1000.0 
        if status != 0:
            print(f"[acados] Abbruch mit Status {status}")
            self._acados_fail += 1
            self._bw_ramp_elapsed = 0.0
            if self._acados_fail > 5:
                print("[acados] To many failures, stopping simulation")
                self.finished = True

                return np.zeros(4)
            return np.array([self.last_f_cmd, *self.last_rpy_cmd])


        # Saving planned trajectory points and drone trajectory for visualizations 
        planned_traj = np.zeros((self.N + 1, 3))
        for j in range(self.N + 1):
            xj = self.acados_ocp_solver.get(j, "x")
            planned_traj[j] = xj[:3]
        self._planned_traj = planned_traj
        self._drone_traj.append(obs["pos"])

        # Get the control commands from the first stage
        x1 = self.acados_ocp_solver.get(1, "x")
        cmd = x1[10:14]
        self.last_theta = x1[14]
        self.last_vtheta = x1[15]
        return cmd

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self.finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0

    def update_trajectory(self, obs: dict[str, NDArray[np.floating]] | None = None):
        """Update the trajectory waypoints and splines based on the current gate and obstacle positions.

        Args:
            obs: The current observation dictionary containing gate and obstacle positions.
        """
        for i, pos in enumerate(obs["gates_pos"]):
            # move the waypoints to the current gate positions
            idx = self._gate_to_wp_index.get(i)
            self._waypoints[idx] = pos
            # some special cases for certain waypoints
            if i == 0:
                self._waypoints[idx + 1] = [
                    obs["obstacles_pos"][1][0] - 0.3,
                    obs["obstacles_pos"][1][1],
                    0.75,
                ]
                self._waypoints[idx + 2] = [
                    obs["obstacles_pos"][1][0],
                    obs["obstacles_pos"][1][1] - 0.4,
                    0.85,
                ]
            if i == 2:
                self._waypoints[idx + 1] = [pos[0] - 0.21, pos[1] + 0.21, pos[2] + 0.18]
            if i == 3 and not obs["target_gate"] == 3:
                if obs["obstacles_pos"][3][0] < -0.45:  
                    self._waypoints[idx - 1] = [
                        obs["obstacles_pos"][3][0] + 0.2,
                        obs["obstacles_pos"][3][1],
                        1.1,
                    ]
                else:
                    self._waypoints[idx - 1] = [
                        obs["obstacles_pos"][3][0] - 0.2,
                        obs["obstacles_pos"][3][1],
                        1.1,
                    ]
        changed = not np.allclose(self._waypoints, self._prev_waypoints)

        # update the cubic spline representation of the trajectory
        seg_lens = np.linalg.norm(np.diff(self._waypoints, axis=0), axis=1)
        s_grid = np.hstack(([0.0], np.cumsum(seg_lens)))  
        self.s_total = float(s_grid[-1])
        cs_x = CubicSpline(s_grid, self._waypoints[:, 0])
        cs_y = CubicSpline(s_grid, self._waypoints[:, 1])
        cs_z = CubicSpline(s_grid, self._waypoints[:, 2])
        self._gate_s = {i: s_grid[idx] for i, idx in self._gate_to_wp_index.items()}
        self._cs_x, self._cs_y, self._cs_z = cs_x, cs_y, cs_z
        vis_s = np.linspace(0.0, self.s_total, 200, endpoint=False)
        self.traj_points = np.column_stack((cs_x(vis_s), cs_y(vis_s), cs_z(vis_s)))

        # Update waypoint
        if changed:
            self.vis_s = vis_s
            dists_new  = np.linalg.norm(self.traj_points - obs["pos"], axis=1)
            self._prev_idx_min_vis = int(np.argmin(dists_new))
            self._prev_waypoints = self._waypoints.copy()   

    def get_trajectory(self) -> NDArray[np.floating]:
        """Get the trajectory points."""
        return self.traj_points

    def get_planned_trajectory(self) -> NDArray[np.floating]:
        """Return the list of 3D waypoints (N+1) that the MPC currently plans."""
        return self._planned_traj

    def get_drone_trajectory(self) -> NDArray[np.floating]:
        """Return the drone trajectory (N+1) that the MPC currently plans."""
        if len(self._drone_traj) == 0:
            return np.empty((0, 3))
        return np.array(self._drone_traj)

    def get_tunnel_regions(self) -> np.ndarray:
        """Return the active tunnel polygons (N×4×3) for each stage of the MPC horizon.

        Uses the stored self._stage_centers, self._stage_nb, and self._stage_wh from compute_control,
        without calling tunnel_bounds again.
        """
        regions = []
        if self._tunnel_centers is None or len(self._tunnel_centers) < self.N:
            raise ValueError("Call compute_control() first to populate stage data.")
        
        # Loop through each stage and compute the tunnel corners
        for j in range(self.N):
            c    = self._tunnel_centers[j]
            n, b = self._tunnel_nb[j]
            w, h = self._tunnel_wh[j]

            w2 = w / 2.0
            h2 = h / 2.0

            corner0 = c +  w2 * n +  h2 * b
            corner1 = c +  w2 * n -  h2 * b
            corner2 = c -  w2 * n -  h2 * b
            corner3 = c -  w2 * n +  h2 * b
            regions.append(np.vstack((corner0, corner1, corner2, corner3)))

        return np.array(regions)

    def get_waypoints(self) -> NDArray[np.floating]:
        """Get the waypoints of the trajectory."""
        return self._waypoints
    
    def get_last_solve_ms(self) -> float:
        """Get the time taken for the last OCP solver solve call in milliseconds."""
        return self._last_solve_ms