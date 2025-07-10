"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import json
import time
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import MX, cos, sin, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.debug_utils import get_logger, LOGDIR

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = get_logger()


MPCC_CFG = dict(
    QC=10,
    QL=40,
    MU=10,
    DVTHETA_MAX=1.6,
    N=20,
    T_HORIZON=0.7,

    ALPHA_INTERP=0.2,  # smoothing factor for tunnel width interpolation: 0=no movement, 1=full shift

    RAMP_TIME=0.7,
    BW_RAMP=0.3,  # ramp time for barrier weight‚
    BARRIER_WEIGHT = 10, 
    
    TUNNEL_WIDTH = 0.4,  # nominal tunnel width
    TUNNEL_WIDTH_GATE = 0.15,  # nominal tunnel height
    NARROW_DIST = 0.7,      # distance (m) at which tunnel starts to narrow
    GATE_FLAT_DIST = 0.1,  # distance (m) from gate at which tunnel width remains at gate size
    
    R_VTHETA = 1.0,        # quadratic penalty on vtheta

    REG_THRUST = 1.0e-2,
    REG_INPUTS = 1.0e-1,

    OBSTACLE_RADIUS = [0.13, 0.23, 0.1, 0.1],
)




def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "lsy_example_mpc"

    # Define Gravitational Acceleration
    GRAVITY = 9.806

    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    """Model setting"""
    # define basic variables in state and input vector
    px = MX.sym("px")  # 0
    py = MX.sym("py")  # 1
    pz = MX.sym("pz")  # 2
    vx = MX.sym("vx")  # 3
    vy = MX.sym("vy")  # 4
    vz = MX.sym("vz")  # 5
    roll = MX.sym("r")  # 6
    pitch = MX.sym("p")  # 7
    yaw = MX.sym("y")  # 8
    f_collective = MX.sym("f_collective")

    f_collective_cmd = MX.sym("f_collective_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")

    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")

    # MPCC extend
    theta = MX.sym("theta")
    vtheta = MX.sym("vtheta")
    dvtheta_cmd = MX.sym("dvtheta_cmd")


    p = MX.sym("p", 18)
    px_r, py_r, pz_r = p[0], p[1], p[2]
    tx,   ty,   tz   = p[3], p[4], p[5]
    qc,   ql,   mu   = p[6], p[7], p[8]
    nx, ny, nz = p[9],  p[10], p[11]
    bx, by, bz = p[12], p[13], p[14]
    w_half     = p[15]
    h_half     = p[16]
    w_bar      = p[17] 




    # define state and input vector
    states = vertcat(
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        roll,
        pitch,
        yaw,
        f_collective,
        f_collective_cmd,
        r_cmd,
        p_cmd,
        y_cmd,
        theta,
        vtheta,
    )
    inputs = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd, dvtheta_cmd)
    droll= params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd
    dpitch =  params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd
    dyaw = 10.0 * (f_collective_cmd - f_collective)
    # Define nonlinear system dynamics
    f = vertcat(
        vx,
        vy,
        vz,
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)),
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)),
        (params_acc[0] * f_collective + params_acc[1]) * cos(roll) * cos(pitch) - GRAVITY,
        params_roll_rate[0] * roll + params_roll_rate[1] * r_cmd,
        droll,
        dpitch,
        dyaw,
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd,
        vtheta,
        dvtheta_cmd,
    )

    # --- MPCC error terms -------------------------------------------------
    pos = vertcat(px, py, pz)
    pref = vertcat(px_r, py_r, pz_r)
    t_vec = vertcat(tx, ty, tz)  # assumed already normalised outside
    delta = pos - pref

    lag_err = delta.T @ t_vec  # scalar
    cont_err = delta - lag_err * t_vec  # 3‑D vector

    ec_sq = cont_err.T @ cont_err  # ‖e_c‖²
    el_sq = lag_err**2  # e_l²
    # --- Extended MPCC++ Cost --------------------------------------------------
    R_vth   = MPCC_CFG["R_VTHETA"]
    
    n_vec = vertcat(nx, ny, nz)
    b_vec = vertcat(bx, by, bz)

    g_n_pos = n_vec.T @ delta - w_half        #  (p‑c)·n ≤  w/2
    g_n_neg = -n_vec.T @ delta - w_half
    g_b_pos = b_vec.T @ delta - h_half        #  (p‑c)·b ≤  h/2
    g_b_neg = -b_vec.T @ delta - h_half
    
    
    
    
    alpha = 100.0
    bar = MX.log(1 + MX.exp(alpha*(g_n_pos))) + MX.log(1 + MX.exp(alpha*(g_n_neg))) \
        + MX.log(1 + MX.exp(alpha*(g_b_pos))) + MX.log(1 + MX.exp(alpha*(g_b_neg)))
    
    R_input = MPCC_CFG["REG_INPUTS"]
    R_thrust = MPCC_CFG["REG_THRUST"]

    stage_cost = (
        qc * ec_sq
        + ql * el_sq
        + R_vth * dvtheta_cmd**2           
        - mu * vtheta                
        + R_input * (dr_cmd**2 + dp_cmd**2 + dy_cmd**2)
        + R_thrust * (df_cmd**2)
        + w_bar * bar
    )





    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs
    model.cost_expr_ext_cost = stage_cost  # stage cost
    model.p = p  # expose parameters to the OCP
    model.con_h_expr = None #con_h      # nonlinear ≤0 constraints
    return model


def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model

    # Get Dimensions
    nx = model.x.rows()

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "EXTERNAL"

    # Tell acados how many parameters each stage has
    ocp.dims.np = 18
    # default parameter vector (will be overwritten at run time)
    ocp.parameter_values = np.zeros(18)


    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13, 15])

    lbx = np.array([0.1, 0.1, -1.3, -1.3, -1.3, -MPCC_CFG["DVTHETA_MAX"]])
    ubx = np.array([0.65, 0.65,  1.3,  1.3,  1.3, MPCC_CFG["DVTHETA_MAX"]])
    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx


    # ---- tunnel constraints via BGH + soft slack -----------------
    #ocp.constraints.constr_type = None #"BGH"
    #ocp.constraints.nh      = 4
    #ocp.constraints.expr_h = None #model.con_h_expr
    #ocp.constraints.lh = -1.0e9 * np.ones(4)   # large negative value instead of -inf (JSON safe)
    #ocp.constraints.uh = np.zeros(4)
    #ocp.constraints.idxsh = np.array([], dtype=int)   # no soft slack, Barrier only

    # ----------------------------------------------------------
    #   TURN OFF ALL path-inequality constraints (h-constraints)
    # ----------------------------------------------------------
    ocp.constraints.nh     = 0                    # Dimension 0
    ocp.constraints.expr_h = None                # nichts mehr verknüpft
    ocp.constraints.lh     = np.zeros((0,))      # leere NumPy-Arrays
    ocp.constraints.uh     = np.zeros((0,))
    # idxsh darf leer bleiben


    # Set Input Constraints
    # ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0. -10.0])
    # ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0])
    # ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Smooth progress acceleration – limit Δvθ
    dvtheta_max = MPCC_CFG["DVTHETA_MAX"]  # m/s², was unlimited
    ocp.constraints.lbu = np.array([-dvtheta_max])
    ocp.constraints.ubu = np.array([dvtheta_max])
    ocp.constraints.idxbu = np.array([4])  # position of dvtheta_cmd in input vector

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # GAUSqcS_NEWTON
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.tol = 1e-7        
    ocp.solver_options.tf = 1.0


    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.nlp_solver_ext_qp_res = 1

    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.solver_options.qp_solver_warm_start = 0
    ocp.solver_options.qp_solver_iter_max = 1000

    ocp.solver_options.regularize_method  = "CONVEXIFY"
    ocp.solver_options.globalization_line_search_use_sufficient_descent = 1


    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file= "lsy_example_mpc.json", verbose=verbose)


    return acados_ocp_solver, ocp


class MPController(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, PARAM_DICT: dict[str, float] | None = None):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            inf  go: Additional environment information from the reset.
            config: The configuration of the environment.
        """  
        super().__init__(obs, info, config)
        if PARAM_DICT is not None:
            MPCC_CFG.update(PARAM_DICT)
        self.freq = config.env.freq
        self._tick = 0
        self._save_tick_pre_gate = 0
        self._save_tick_post_gate = 0
        self._update_tick_post_gate = False
        self._update_tick_pre_gate = False
        self._target_gate_pos = obs["gates_pos"][obs["target_gate"]]  # 3‑D np.array
        self.d_gate = 0.1
        gates_pos  = obs["gates_pos"]           # shape (4,3)
        gates_quat = obs["gates_quat"]          # shape (4,4)  (w,x,y,z) or (x,y,z,w) ?
        pre_post   = [pre_post_points(gates_pos[i], gates_quat[i], self.d_gate)
                      for i in range(len(gates_pos))]

        pre_gate0, post_gate0 = pre_post[0]
        pre_gate1, post_gate1 = pre_post[1]
        pre_gate2, post_gate2 = pre_post[2]
        pre_gate3, post_gate3 = pre_post[3]

        # Same waypoints as in the trajectory controller. Determined by trial and error.
        waypoints = np.array(
            [
                [1.0896959, 1.4088244, MPCC_CFG["TUNNEL_WIDTH"] / 2],
                [0.95, 1.0, 0.3],
                [0.7, 0.1, 0.4],
                pre_gate0,
                gates_pos[0],     # gate0 centre
                post_gate0,
                [0.4, -1.4, 0.85],
                pre_gate1,
                gates_pos[1],     # gate1 centre
                post_gate1,
                [0.5, 0.1, 0.8],
                pre_gate2,
                gates_pos[2],     # gate2 centre
                post_gate2,
                [-0.1, gates_pos[2][1]+0.2, 0.7],
                [-0.2, gates_pos[2][1]+0.17, 1.0],
                [-0.3, 0.4, 1.1],
                pre_gate3,
                gates_pos[3],     # gate3 centre
                post_gate3,
                [-0.5, -2, 1.11],
                [-0.5, -6, 1.11]
            ]
        )

        self._waypoints = waypoints
        # Map each gate index to its waypoint index (centre of gate in list)
        self._gate_to_wp_index = {
            0: 4,
            1: 8,
            2: 12,
            3: 18,
        }
        #-----------------------------------------------------------------
        # Arc‑length parametrisation of the reference path  (no wall‑clock time)
        # ------------------------------------------------------------------
        # 1. Spline in arclength domain s  (units = metres)
        seg_lens = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        s_grid = np.hstack(([0.0], np.cumsum(seg_lens)))  # shape (K,)
        self.s_total = float(s_grid[-1])

        cs_x = CubicSpline(s_grid, waypoints[:, 0])
        cs_y = CubicSpline(s_grid, waypoints[:, 1])
        cs_z = CubicSpline(s_grid, waypoints[:, 2])

        # Store splines for fast evaluation
        self._cs_x, self._cs_y, self._cs_z = cs_x, cs_y, cs_z
        # Keep medium‑resolution points only for visualisation
        vis_s = np.linspace(0.0, self.s_total, 200, endpoint=False)
        self.traj_points = np.column_stack((cs_x(vis_s), cs_y(vis_s), cs_z(vis_s)))

        # ------------------------------------------------------------------
        # MPC settings (unchanged)
        # ------------------------------------------------------------------
        self.N = MPCC_CFG["N"]  # number of steps in the MPC
        self.T_HORIZON = MPCC_CFG["T_HORIZON"]  # time span of the MPC
        self.dt = self.T_HORIZON / self.N

        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)
        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.last_theta = 0.3
        self.last_vtheta = 0
        self.config = config
        self.finished = False
        self._gate_idx = 0
        self._target = True
        self.start = True
        self._ref_point = np.zeros(3)
        self.safe_pos = np.zeros(3)  
        self._planned_traj = np.zeros((self.N + 1, 3))
        self._pos_on_path = None  
        self._prev_waypoints = self._waypoints.copy()
        self._bw_ramp_elapsed = 0.0
        
        # Store predicted theta trajectory for reference in next iteration
        self._predicted_theta = np.zeros(self.N + 1)
        # storage for gate constraint polygons per MPC iteration
        self._gate_regions_arr: np.ndarray | None = None
        def _pos_on_path(s: float) -> np.ndarray:
            s_wrap = s % self.s_total
            return np.array([self._cs_x(s_wrap),
                            self._cs_y(s_wrap),
                            self._cs_z(s_wrap)])
        self.pos_on_path = _pos_on_path          # <-- expose helper
        
        # Precompute tunnel geometry at high resolution along full path
        self.vis_s = vis_s  # array of 200 sample arclengths
        self.tunnel_cache = []
        self.w_nom = MPCC_CFG["TUNNEL_WIDTH"]
        self.h_nom = MPCC_CFG["TUNNEL_WIDTH"]
        for s in vis_s:
            c, n, b, w, h = tunnel_bounds(self.pos_on_path, s, w_nom=self.w_nom, h_nom=self.h_nom)
            self.tunnel_cache.append((c, n, b, w, h))
        self._target_gate_pos = obs["gates_pos"][obs["target_gate"]]             # (für Debug/Plot)
        # flag to generate a straight tunnel to the gate once
        self._use_line_tunnel = False
        # Placeholder for dynamic gate information (position + orientation quaternion)
        self.gate_pos: np.ndarray | None = None
        self.gate_quat: np.ndarray | None = None
        # Gate opening size (full width/height) in metres
        self.gate_width: float = 0.30   # reduced width of gate opening
        self.gate_height: float = 0.30  # reduced height of gate opening
        self.gate_depth: float = 0.3   # thickness along path/tangent (m)
        self._target_gate = 0
        self._current_gate = obs["gates_pos"][obs["target_gate"]]
        self._current_gate = np.array([0.0, 0.0, 0.0])  # Placeholder for current gate position
        self._planning_points = np.zeros((self.N + 1, 3))  # Store the planned trajectory points

        self._init_obs = np.asarray(obs["obstacles_pos"])

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
        # Optional debug print:
        # print("Target gate:", target_gate_pos)
        # finish when progress (theta) reaches the end of the spline trajectory
        self._planning_points = []  # Reset the planning points for each step
        # reset per‑stage tunnel sizes (w, h) so get_tunnel_regions() can use them later
        self._stage_wh: list[tuple[float, float]] = []
        self._stage_nb: list[tuple[np.ndarray, np.ndarray]] = []
        alpha_interp = MPCC_CFG['ALPHA_INTERP']  # smoothing factor: 0=no movement, 1=full shift
        prev_centres = getattr(self, '_prev_stage_centres', None)
        self._stage_centres: list[np.ndarray] = []


        if self._tick == 0:             # nur am allerersten Aufruf
            self._dump_initial_state(obs)
        self.update_trajectorie(obs["gates_pos"], obs["gates_quat"])
        # Reset tunnel target to current gate for each call
        self._target_gate_pos = obs["gates_pos"][int(obs["target_gate"])]
        if self._bw_ramp_elapsed is not None:
            self._bw_ramp_elapsed += self.dt



        if False:
            print('Finished')
            self.finished = True

        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians
        pos = obs["pos"]
        # ------------------------------------------------------------
        # Gate frame logic removed; always use tunnel geometry below.

        # --- proje1t current position onto path to get up‑to‑date arclength s_cur
        # Use the medium‑resolution path sample self.vis_s (len≈200)
        traj_points_vis = self.traj_points          # (200,3) cached during init
        dists_vis = np.linalg.norm(traj_points_vis - pos, axis=1)
        idx_min_vis = int(np.argmin(dists_vis))
        s_cur = self.vis_s[idx_min_vis]             # arclength of closest point

        
        xcurrent = np.zeros(16)
        xcurrent[:3] = obs["pos"]
        xcurrent[3:6] = obs["vel"]
        xcurrent[6:9] = rpy          # roll, pitch, yaw
        xcurrent[9]   = self.last_f_collective
        xcurrent[10]  = self.last_f_cmd
        xcurrent[11:14] = self.last_rpy_cmd
        xcurrent[14]  = self.last_theta
        xcurrent[15]  = self.last_vtheta



        try:
            vtheta_pred = np.zeros(self.N)
            for j in range(1, self.N+1):
                xj = self.acados_ocp_solver.get(j, "x")
                vtheta_pred[j-1] = float(xj[15])
            theta_pred = np.zeros(self.N + 1)
            theta_pred[0] = s_cur   # start of prediction = current position on path
            for j in range(self.N):
                theta_pred[j+1] = theta_pred[j] + vtheta_pred[j] * self.dt
        except Exception:
            # Erster Aufruf: noch keine vorherigen Prädiktionen
            theta_pred = self.last_theta + self.last_vtheta * np.linspace(0.0, self.T_HORIZON, self.N + 1)
            theta_pred[0] = s_cur

        # if self._tick % 20 == 0:
        #     print(f"θ={self.last_theta:6.2f}  vθ={self.last_vtheta:4.2f}")
        #     True

        # Store predicted theta trajectory so that get_tunnel_regions()
        # can draw the stage‑wise tunnel rectangles for the next visualization
        self._predicted_theta = theta_pred.copy()
        # save also current center for visualization
        self._tunnel_center_now = self.pos_on_path(s_cur)
        
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)
        
        # Standard tunnel selection for all stages; gate logic removed.
        for j in range(self.N):

            s_ref = theta_pred[j]
            pref = self.pos_on_path(s_ref)
            self._ref_point = pref
            pref_next = self.pos_on_path(s_ref + 0.001)
            tangent = unit(pref_next - pref)
            if self._bw_ramp_elapsed is not None:
                ramp = min(1.0, self._bw_ramp_elapsed / MPCC_CFG["BW_RAMP"])
                bw_val = MPCC_CFG["BARRIER_WEIGHT"] * ramp**2
            else:
                bw_val = MPCC_CFG["BARRIER_WEIGHT"]
            # Compute Euclidean distance to gate centre for symmetric tunnel sizing
            gate_pos = obs["gates_pos"][int(obs["target_gate"])]
            dist_gate = np.linalg.norm(pref - gate_pos)

            # Base (far) and minimum (at‑gate) tunnel sizes
            w_far   = MPCC_CFG["TUNNEL_WIDTH"]
            h_far   = MPCC_CFG["TUNNEL_WIDTH"]
            w_gate  = MPCC_CFG["TUNNEL_WIDTH_GATE"]
            h_gate  = MPCC_CFG["TUNNEL_WIDTH_GATE"]

            flat_dist = MPCC_CFG["GATE_FLAT_DIST"]
            narrow_dist = MPCC_CFG["NARROW_DIST"]
            if dist_gate > narrow_dist:
                # far from gate: full tunnel width
                w_stage = w_far
                h_stage = h_far
            elif dist_gate > flat_dist:
                # narrowing region: interpolate between wide and gate widths
                ratio = (dist_gate - flat_dist) / (narrow_dist - flat_dist)
                w_stage = ratio * w_far + (1.0 - ratio) * w_gate
                h_stage = ratio * h_far + (1.0 - ratio) * h_gate
            else:
                # within flat region near the gate: constant gate width
                w_stage = w_gate
                h_stage = h_gate

            # Compute tunnel orientation with the stage‑specific width/height
            c, n, b, w, h = tunnel_bounds(
                self.pos_on_path,
                s_ref,
                w_nom=w_stage,
                h_nom=h_stage
            )

            self._stage_nb.append((n, b))
            # Shift the reference centre laterally so that every obstacle is
            # at least (w_stage/2 + margin) away – without changing the tunnel width.
            curent_obs = np.asarray(obs["obstacles_pos"])
            init_obs = self._init_obs
            obs_arr = np.zeros_like(init_obs)
            mask = np.array([not np.allclose(cur, init) for cur, init in zip(curent_obs, init_obs)])
            obs_arr[mask] = curent_obs[mask]

            pref_shift, _ = shrink_side_for_obstacles(pref ,n, w_stage, obs_arr)
            # apply interpolation to avoid jumps
            if prev_centres is not None and len(prev_centres) > j:
                prev_c = prev_centres[j]
            else:
                prev_c = pref
            smoothed_c = prev_c + alpha_interp * (pref_shift - prev_c)
            self._stage_wh.append((w, h))
            self._stage_centres.append(smoothed_c)


            # Parameter ramp-up at start
            ramp_start = min(1.0, self._tick * self.dt / MPCC_CFG["RAMP_TIME"])
            qc_val = MPCC_CFG["QC"] * (0.5 + 0.5 * ramp_start)
            ql_val = MPCC_CFG["QL"] * (0.5 + 0.5 * ramp_start)
            mu_val = MPCC_CFG["MU"] * ramp_start

            p_vec = np.concatenate([
                smoothed_c,
                tangent,
                [qc_val, ql_val, mu_val],
                n,
                b,
                [w/2, h/2, bw_val]
            ])
            self.acados_ocp_solver.set(j, "p", p_vec)
            self._planning_points.append(smoothed_c)  # Store the planned trajectory points
        self._prev_stage_centres = list(self._stage_centres)

        self.acados_ocp_solver.options_set("qp_warm_start", 2)    # 2 = full

        tic = time.perf_counter()
        status = self.acados_ocp_solver.solve()
        toc = time.perf_counter()

        if status != 0:
            print(f"[acados] Abbruch mit Status {status}")
            self._dump_failure(status, tic, toc)
            return np.array([self.last_f_cmd, *self.last_rpy_cmd])


        x1 = self.acados_ocp_solver.get(1, "x")
        u1 = self.acados_ocp_solver.get(1, "u")
        
        # input_names = ["df_cmd", "dr_cmd", "dp_cmd", "dy_cmd", "dv_theta_cmd"]
        # for name, value in zip(input_names, u1):
        #     print(f"Input {name}: {value:.7f}")

        cmd = x1[10:14]
        self.last_theta = x1[14]
        self.last_vtheta = x1[15]
  
        planned_traj = np.zeros((self.N + 1, 3))
        for j in range(self.N + 1):
            xj = self.acados_ocp_solver.get(j, "x")
            planned_traj[j] = xj[:3]   # first three state entries: px, py, pz
        self._planned_traj = planned_traj
        
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

    def get_trajectory(self) -> NDArray[np.floating]:
        """Get the trajectory points."""
        return self.traj_points

    def get_ref_point(self) -> NDArray[np.floating]:
        """Get the reference point."""
        return self._ref_point
        
    def get_planned_trajectory(self) -> NDArray[np.floating]:
        """Return the list of 3‑D waypoints (N+1) that the MPC currently plans."""
        return self._planned_traj
    
    def get_pos_on_path(self) -> NDArray[np.floating]:
        """Get the position on the reference path for a given arclength s."""
        return self._pos_on_path
    
    def get_tunnel_regions(self) -> np.ndarray:
        """
        Returns the currently active position-constraints (rectangles) for every
        stage of the MPC horizon. Always returns the regular tunnel rectangles
        along the predicted theta trajectory (shape (N, 4, 3)).
        """
        theta_vals = getattr(self, "_predicted_theta", None)
        if theta_vals is None or len(theta_vals) < self.N:
            raise ValueError(
                "Predicted theta trajectory not available. Call compute_control() first."
            )
        regions = []
        for j in range(self.N):
            s_ref = theta_vals[j]
            # orientation vectors always freshly recomputed
            if hasattr(self, "_stage_centres") and j < len(self._stage_centres):
                c = self._stage_centres[j]
                n, b = self._stage_nb[j]
            else:
                c, n, b, _, _ = tunnel_bounds(
                    self.pos_on_path, s_ref,
                    w_nom=self.w_nom, h_nom=self.h_nom
                )
            # width/height: take the values actually used during compute_control
            if hasattr(self, "_stage_wh") and j < len(self._stage_wh):
                w, h = self._stage_wh[j]
            else:
                w, h = self.w_nom, self.h_nom
            w_half = w / 2.0
            h_half = h / 2.0
            corner0 = c + w_half * n + h_half * b
            corner1 = c + w_half * n - h_half * b
            corner2 = c - w_half * n - h_half * b
            corner3 = c - w_half * n + h_half * b
            regions.append(np.vstack((corner0, corner1, corner2, corner3)))
        return np.array(regions)

    def get_waypoints(self) -> NDArray[np.floating]:
        """Get the waypoints of the trajectory."""
        return self._waypoints
    
    def get_planning_points(self) -> NDArray[np.floating]:
        """Get the reference points of the trajectory."""
        return self._planning_points
    
    def _dump_initial_state(self, obs):
        # log.info("===== INITIAL STATE DUMP =====")
        # log.debug(json.dumps({
        #     "xcurrent": obs["pos"].tolist() + obs["vel"].tolist(),
        #     "params":   MPCC_CFG,
        #     "box_lims": {
        #         "vtheta_max": float(self.ocp.constraints.ubx[-1]),
        #         "dvtheta_max": float(self.ocp.constraints.ubu[0]),
        #     }
        # }, indent=2))
        True

    def _dump_failure(self, status, tic, toc):
        it_tot = int(self.acados_ocp_solver.get_stats("sqp_iter"))

        # ---- KKT robust auslesen ------------------------------------------
        try:                                      # v0.2 / v0.3
            kkt = float(self.acados_ocp_solver.get_stats("stat_res_kkt")[it_tot])
        except ValueError:
            stats = self.acados_ocp_solver.get_stats("statistics")  # (m, n)
            if stats.ndim == 2 and stats.shape[1] >= 5:
                row = min(it_tot, stats.shape[0]-1)   # Clamp!
                kkt = float(stats[row, 4])            # Spalte 4 = KKT
            else:                                     # Fallback
                kkt = float(self.acados_ocp_solver.get_stats("res_stat_all")[0])

        # log.error(f"Solver failed | status={status} | SQP it={it_tot} "
        #         f"| KKT={kkt:.2e} | t={toc-tic:.3f}s")

        # ---- Dump ---------------------------------------------------------
        x_seq, u_seq = [], []
        for j in range(self.N+1):
            try:
                x_seq.append(self.acados_ocp_solver.get(j, "x").tolist())
                if j < self.N:
                    u_seq.append(self.acados_ocp_solver.get(j, "u").tolist())
            except RuntimeError:          # falls Zugriff nach Abbruch scheitert
                break

        dump_dir = (LOGDIR / "fail"); dump_dir.mkdir(parents=True, exist_ok=True)
        path = dump_dir / f"tick{self._tick:04d}.json"
        path.write_text(json.dumps({
            "status": status, "iter": it_tot, "kkt": kkt,
            "x0": x_seq[0] if x_seq else None,
            "x_seq": x_seq, "u_seq": u_seq,
        }, indent=1))
        #log.error(f"Dump saved → {path}")

    def update_trajectorie(self, gate_pos: NDArray[np.floating], gate_quat: NDArray[np.floating]):
        for i, (pos, quat) in enumerate(zip(gate_pos, gate_quat)):
            idx = self._gate_to_wp_index.get(i)
            pre,post = pre_post_points(pos, quat, self.d_gate)            
            self._waypoints[idx-1] = pre
            self._waypoints[idx] = pos
            self._waypoints[idx+1] = post
            if i == 2:
                self._waypoints[idx+2][1] = pos[1] + 0.2
                self._waypoints[idx+3][1] = pos[1] + 0.17
            # Optional: falls du auch mit gate_quat etwas machen willst, hier weiterverarbeiten

        # ---- Detect waypoint changes ---------------------------------------
        changed = not np.allclose(self._waypoints, self._prev_waypoints)
        if changed:
            self._bw_ramp_elapsed = 0.0       # start BW‑ramp timer
            self._prev_waypoints = self._waypoints.copy()

        seg_lens = np.linalg.norm(np.diff(self._waypoints, axis=0), axis=1)
        s_grid = np.hstack(([0.0], np.cumsum(seg_lens)))  # shape (K,)
        self.s_total = float(s_grid[-1])

        cs_x = CubicSpline(s_grid, self._waypoints[:, 0])
        cs_y = CubicSpline(s_grid, self._waypoints[:, 1])
        cs_z = CubicSpline(s_grid, self._waypoints[:, 2])

        # update arclength position of every gate centre
        self._gate_s = {i: s_grid[idx] for i, idx in self._gate_to_wp_index.items()}
        # Store splines for fast evaluation
        self._cs_x, self._cs_y, self._cs_z = cs_x, cs_y, cs_z
        # Keep medium‑resolution points only for visualisation
        vis_s = np.linspace(0.0, self.s_total, 200, endpoint=False)
        self.traj_points = np.column_stack((cs_x(vis_s), cs_y(vis_s), cs_z(vis_s)))


    # get_gate_regions removed: no longer used.

def tunnel_bounds(pos_on_path, s: float,
                w_nom: float = 0.4, h_nom: float = 0.4):
    """Berechne Center-, Normal-, Binormal-Vektor & Breite/Höhe."""
    c      = pos_on_path(s)
    c_next = pos_on_path(s + 0.01)
    t      = c_next - c;  t /= np.linalg.norm(t) + 1e-9
    n      = np.array([-t[1], t[0], 0.0])
    n /= np.linalg.norm(n) + 1e-9
    b      = np.cross(t, n)               
    b /= np.linalg.norm(b) + 1e-9
    return c, n, b, w_nom, h_nom

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

def pre_post_points(gate_pos: np.ndarray,
                    gate_quat: np.ndarray,
                    dist: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute points before and after a gate along its local normal.
    gate_quat is assumed in [x, y, z, w] order.
    """
    # build rotation from quaternion
    rot = R.from_quat(gate_quat)
    # local normal along gate's Z-axis
    normal = rot.apply(np.array([0.0, 1.0, 0.0]))
    # normalize to unit length
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    pre = gate_pos - dist * normal
    post = gate_pos + dist * normal
    return pre, post

def move_for_obstacles(
    pref: np.ndarray,
    w_nom: float,
    obstacles: np.ndarray,
    n_vec: np.ndarray = None,
    margin: float = 0.05,
    max_shift: float = 0.25,
) -> np.ndarray:
    """
    Move the tunnel centre so that every obstacle in `obstacles`
    is at least `margin` away **inside** the nominal full width `w_nom`.
    The check is performed along the normal direction `n_vec` if provided,
    but the shift is still along the direction from pref to obs (in xy).
    Returns
        c_shifted : new centre (3,)
    """
    if obstacles.size == 0:
        return pref

    c_shifted = pref.copy()

    for obs in obstacles:
        # Project obstacle onto normal direction if n_vec is given, else use xy distance
        obs_vec = obs[:2] - c_shifted[:2]
        dist = np.linalg.norm(obs_vec)
        thr = w_nom / 2.0 + margin
        if dist < thr:
            direction_xy = (c_shifted[:2] - obs[:2]) / (dist + 1e-9)
            required_shift = thr - dist
            shift = np.clip(required_shift, 0, max_shift)
            c_shifted[:2] = c_shifted[:2] + shift * direction_xy

    return c_shifted

def shrink_side_for_obstacles(
    pref: np.ndarray,
    n_vec: np.ndarray,
    w_nom: float,
    obstacles: np.ndarray,
    *,
    look_ahead: float = 0.40  ,     # wie weit „vorne“ ein Hindernis (entlang t) noch gilt
    look_behind: float = 0.18,    # wie weit „hinten“ es noch gilt
    max_shift: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Verschiebt den Tunnel-Mittelpunkt seitlich (XY), sodass jeder Hindernis-Kreis
    mindestens w_nom/2 Abstand von der Tunnelwand hat **und** der Versatz
    sofort zurückgeht, sobald wir das Hindernis hinter uns lassen.

    Args
    ----
    pref : (3,) aktueller Tunnel-Mittelpunkt
    n_vec: (3,) Einheitsvektor seitlich (Breite)
    w_nom: nominale Tunnelbreite (m)
    obstacles: (K,3) Hindernismittelpunkte (Z-Koord. wird ignoriert)
    look_ahead : Positives Fenster (m) vor der Drohne, in dem Hindernisse
                 berücksichtigt werden.
    look_behind: Kleines Fenster (m) hinter der Drohne, in dem Hindernisse
                 noch berücksichtigt werden (Hysterese).
    max_shift  : optionale Begrenzung des Gesamt-Shifts (m)

    Returns
    -------
    new_center : (3,) verschobener Tunnel-Mittelpunkt
    w_nom      : unveränderte Tunnelbreite (wird für Call-Site-Kompatibilität
                 unverändert zurückgegeben)
    """
    # ---------------- Basis-Vorbereitung -------------------------------
    w_half = w_nom / 2.0
    pref_xy = pref[:2]

    n_xy = n_vec[:2]
    norm_n = np.linalg.norm(n_xy)
    if norm_n < 1e-9:                       # degenerierter Vektor
        return pref.copy(), w_nom
    n_xy /= norm_n

    # Tangentialer Vektor in der XY-Ebene (rechtshändig zu n_xy)
    t_xy = np.array([n_xy[1], -n_xy[0]])

    # ---------------- Shift-Berechnung ---------------------------------
    shift = 0.0
    radii = MPCC_CFG.get("OBSTACLE_RADIUS", [])
    default_r = 0.10                       # fallback, falls Liste zu kurz

    for i, p in enumerate(obstacles):
        # Platzhalter-Einträge überspringen
        if np.allclose(p, 0.0):
            continue

        p_xy = p[:2]
        r_obs = radii[i] if i < len(radii) else default_r

        # Längsabstand (Vorzeichen: >0 vor uns, <0 hinter uns)
        d_long = float(t_xy @ (p_xy - pref_xy))
        if d_long >  look_ahead + r_obs:
            continue                       # noch zu weit vorne
        if d_long < -look_behind - r_obs:
            continue                       # schon lange überholt

        # Lateraler Abstand
        d_lat = float(n_xy @ (p_xy - pref_xy))
        eff_d = abs(d_lat) - r_obs         # „Rand-zu-Rand“-Abstand

        if eff_d < w_half:                 # Kreis schneidet inneren Bereich
            needed_shift = w_half - eff_d
            # d_lat >0 → Hindernis rechts → Tunnel nach links (neg. n-Richtung)
            shift += -np.sign(d_lat) * needed_shift

    # -------------- Nachbearbeitung ------------------------------------
    if max_shift is not None:
        shift = np.clip(shift, -max_shift, max_shift)

    new_pref_xy = pref_xy + shift * n_xy
    new_center = np.array([new_pref_xy[0], new_pref_xy[1], pref[2]])

    return new_center, w_nom