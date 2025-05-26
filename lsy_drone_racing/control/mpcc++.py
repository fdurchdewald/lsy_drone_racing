"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


# MPCC_CFG = dict(
#     QC=2,
#     QL=8,
#     MU=0.000000000000000001,
#     DVTHETA_MAX=0.4,
#     N=15,
#     T_HORIZON=1.5,
#     RAMP_TIME=2.0,
#     BARRIER_WEIGHT = 10,          # tunnel slack weight
#     OBSTACLE_WEIGHT = 10000         # obstacle slack weight
# )
MPCC_CFG = dict(
    QC=20,
    QL=100,
    MU=0.0001,
    DVTHETA_MAX=2,
    N=5,
    T_HORIZON=0.5,
    RAMP_TIME=2.0,
    BARRIER_WEIGHT = 100,          # tunnel slack weight
    OBSTACLE_WEIGHT = 1000         # obstacle slack weight
)
# define a single vertical cylinder obstacle by its top-center coordinate and radius
OBSTACLE_TOP = [-0.5, 0.5, 2]  # replace with your actual coordinate
OBSTACLE_RADIUS = 0.13



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

    # parameter vector: [pref(3), t(3), qc, ql, mu, pg(3), q_gate,   R_DF, R_DV, Q_OMEGA]
    # 
    p = MX.sym("p", 17)
    # px_r, py_r, pz_r = p[0], p[1], p[2]
    # tx, ty, tz = p[3], p[4], p[5]
    # qc, ql, mu = p[6], p[7], p[8]
    # pgx, pgy, pgz = p[9], p[10], p[11]
    # q_gate = p[12]
    # R_df, R_dv, Q_omega = p[13], p[14], p[15]
    # p-Vektor aufteilen
    px_r, py_r, pz_r = p[0], p[1], p[2]
    tx,   ty,   tz   = p[3], p[4], p[5]
    qc,   ql,   mu   = p[6], p[7], p[8]
    nx, ny, nz   = p[9],  p[10], p[11]
    bx, by, bz   = p[12], p[13], p[14]
    whalf, hhalf = p[15], p[16]
    n_vec = vertcat(nx, ny, nz)
    b_vec = vertcat(bx, by, bz)


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
        params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd,
        params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd,
        10.0 * (f_collective_cmd - f_collective),
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

    #MPCC++
    # ---------- Tunnel soft-barrier ---------- 
    ALPHA  = 8.0   # Steilheit
    LAMBDA = 2.0   # Gewicht
    EPS_H  = 1e-4

    # seitlicher / vertikaler Offset relativ zur Centerline
    e_n = (pos - pref).T @ n_vec         # rechts (+) / links (-)
    e_b = (pos - pref).T @ b_vec         # oben (+) / unten (-)

    h1 = whalf - e_n    # linker Rand
    h2 = whalf + e_n    # rechter Rand
    h3 = hhalf - e_b    # unten
    h4 = hhalf + e_b    # oben

    def soft_barrier(h):
        # clip vermeidet exp(±groß) → Inf
        h_clip = MX.fmax(h, -1.0)          # alles < −1 wird auf −1 begrenzt
        return LAMBDA * MX.log1p(MX.exp(-ALPHA * h_clip))

    barrier_cost = (
        soft_barrier(h1) + soft_barrier(h2) +
        soft_barrier(h3) + soft_barrier(h4)
    )
    print("Symbolic barrier cost shape:", barrier_cost.shape)
    # dbarrier_dpos = MX.gradient(barrier_cost, pos)
    # print("∥∇Barrier∥ symbolic:", np.linalg.norm(np.array(dbarrier_dpos)).round(3))

    # soft-tunnel constraint weight
      # tune this parameter as needed


    # add a small regularization on the inputs and include tunnel soft constraint
    R_reg = 1e-1
    stage_cost = (
        qc * ec_sq
        + ql * el_sq
        - mu * vtheta
        + R_reg * (df_cmd**2 + dr_cmd**2 + dp_cmd**2 + dy_cmd**2 + dvtheta_cmd**2)
    )
    term_cost  = qc*ec_sq + ql*el_sq                 # Terminal ohne Barrier



    model = AcadosModel()
    model.name = model_name
    # tunnel walls h1..h4 and cylindrical obstacle h_obs ≥ 0
    ox, oy, oz = OBSTACLE_TOP
    # horizontal (xy) distance from drone to obstacle center
    h_obs = MX.sqrt((px - ox)**2 + (py - oy)**2) - OBSTACLE_RADIUS
    model.con_h_expr = vertcat(h1, h2, h3, h4, h_obs)
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs
    model.cost_expr_ext_cost = stage_cost  # stage cost
    model.cost_expr_ext_cost_e = term_cost  # terminal cost
    model.p = p  # expose parameters to the OCP
    return model


def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model
    # soft tunnel constraints and obstacle
    ocp.dims.nh = 5
    ocp.constraints.lh = np.zeros(5)                 # h >= 0
    ocp.constraints.uh = 1e7 * np.ones(5)  # large finite bound instead of inf
    ocp.constraints.soft_constraint = "SLACK"         # enable soft constraints
    # separate slack penalties: first 4 for tunnel, last 1 for obstacle
    tunnel_w = MPCC_CFG["BARRIER_WEIGHT"]
    obs_w    = MPCC_CFG["OBSTACLE_WEIGHT"]
    w_soft = np.hstack([tunnel_w * np.ones(4), obs_w])
    ocp.cost.W_soft = np.diag(w_soft)

    # Get Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # Tell acados how many parameters each stage has
    ocp.dims.np = 17
    # default parameter vector (will be overwritten at run time)
    ocp.parameter_values = np.zeros(17)

    # Set state constraints: collective thrust, Euler commands, and virtual progress speed
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13, 15])
    ocp.constraints.lbx   = np.array([0.1, 0.1, -1.57, -1.57, -1.57, 0.0])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.0, 1.0, 1.0, MPCC_CFG["DVTHETA_MAX"]])

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
    ocp.solver_options.hessian_approx = "EXACT"  # GAUSS_NEWTON
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.tol = 1e-3        # require NLP residual < 1e-4

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.nlp_solver_max_iter = 300
    ocp.solver_options.qp_solver_iter_max = 300  # limit QP to 100 iterations

    ocp.solver_options.regularize_method  = "CONVEXIFY"
    ocp.solver_options.line_search_use_sufficient_descent = 1

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc.json", verbose=verbose)


    return acados_ocp_solver, ocp


class MPController(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """  # noqa: D205
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._tick = 0
        self._target_gate_pos = obs["gates_pos"][obs["target_gate"]]  # 3‑D np.array
        # Same waypoints as in the trajectory controller. Determined by trial and error.
        waypoints = np.array(
            [
                [1.0896959, 1.4088244, 0.08456537],
                [0.8, 1.0, 0.2],
                [0.55, -0.3, 0.5],
                [0.2, -1.3, 0.65],
                [1.1, -0.85, 1.1],
                [0.2, 0.5, 0.65],
                [0.0, 1.2, 0.525],
                [0.0, 1.2, 1.1],
                [-0.5, 0.0, 1.1],
                [-0.5, -0.5, 1.1],
                [-0.5, -2, 1.1],
            ]
        )
        # ------------------------------------------------------------------
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
        vis_s = np.linspace(0.0, self.s_total, 200)
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
        self.last_vtheta = 0.0
        self.config = config
        self.finished = False
        self._gate_idx = 0
        self._use_normal_tunnel = True  # use normal tunnel geometry
        self._tunnel_gen = True
        self._target = True
        self.start = True
        self._ref_point = np.zeros(3)
        self.safe_pos = np.zeros(3)  
        # Store predicted theta trajectory for reference in next iteration
        self._predicted_theta = np.zeros(self.N + 1)
        def _pos_on_path(s: float) -> np.ndarray:
            s_wrap = s % self.s_total
            return np.array([self._cs_x(s_wrap),
                            self._cs_y(s_wrap),
                            self._cs_z(s_wrap)])
        self.pos_on_path = _pos_on_path          # <-- expose helper
        self.tunnel_cache: list = []
        self._target_gate_pos = obs["gates_pos"][obs["target_gate"]]             # (für Debug/Plot)
        # flag to generate a straight tunnel to the gate once
        self._use_line_tunnel = False

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
        if False:
            print('Finished')
            self.finished = True
        # detect new gate and enable line tunnel
        if not np.allclose(self._target_gate_pos, obs["gates_pos"][obs["target_gate"]]):
            print('switching to line tunnel')
            self.tunnel_cache = []

            self._target_gate_pos = obs["gates_pos"][obs["target_gate"]]
            self._use_line_tunnel = True
            self.tunnel_cache.clear()
        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians

        xcurrent = np.concatenate(
            (
                obs["pos"],  # 3
                obs["vel"],  # 3
                rpy,  # 3
                np.array([self.last_f_collective, self.last_f_cmd]),  # 2
                self.last_rpy_cmd,  # 3
                np.array([self.last_theta, self.last_vtheta]),  # 2
            )
        )
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        def unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

        # helper: position on the reference path for a given arclength s
        def pos_on_path(s: float) -> np.ndarray:
            s_wrap = s % self.s_total
            return np.array([self._cs_x(s_wrap), self._cs_y(s_wrap), self._cs_z(s_wrap)])

        def pos_on_line(p1: np.ndarray, p2: np.ndarray, s: float) -> np.ndarray:
            """Linear interpolation between two 3D points p1 and p2.

            Args:
                p1, p2: np.ndarray of shape (3,) defining endpoints.
                s: float in [0,1], interpolation parameter.

            Returns:
                3D point on the line segment from p1 to p2.
            """  # noqa: D205
            # clamp s between 0 and 1
            s_clamped = max(0.0, min(1.0, s))
            return (1 - s_clamped) * p1 + s_clamped * p2
        
        theta_pred = self.last_theta + self.last_vtheta * np.linspace(
                0.0, self.T_HORIZON, self.N + 1
            )           # shape (N+1,)
        if self._tick % 20 == 0:
            print(f"θ={self.last_theta:6.2f}  vθ={self.last_vtheta:4.2f}")


        # upon gate change, generate full straight tunnel once

        for j in range(self.N):
            s_ref = theta_pred[j]
            # compute default reference point on path
            pref = self.pos_on_path(s_ref)
            self._ref_point = pref  # store for visualisation
            pref_next = self.pos_on_path(s_ref + 0.001)
            tangent = unit(pref_next - pref)
            step_size = 0.4                                  # z.B. 5 cm
            pref_ahead = pref + step_size * tangent
            # --- Tunnel-Geometrie -------------------------------------------------
            if self._use_line_tunnel:
                p2 = obs["gates_pos"][obs["target_gate"]]
                # normalized parameter along the straight line [0,1]
                s_line = j / (self.N - 1) if self.N > 1 else 0.0
                # build control points array for the spline: from current safe_pos to gate midpoint
                points = np.vstack([self.safe_pos, p2 ,pref_ahead])
                # extend line so that p2 is midpoint between safe_pos and new endpoint
                p3 = 2 * p2 - self.safe_pos
                #c, n, b, w, h = spline_tunnel_bounds(points, s_line, w_nom=0.25, h_nom=0.25)
                c, n, b, w, h = line_tunnel_bounds(self.safe_pos, p3, s_line, w_nom=0.15, h_nom=0.15)
                self._use_normal_tunnel = False
                if self._gate_idx != obs["target_gate"]:
                    self._gate_idx = obs["target_gate"]
                    self._use_normal_tunnel = True
                    self._use_line_tunnel = False
            if self._use_normal_tunnel:
                self.safe_pos = obs["pos"]  # save current position for tunnel generation
                c, n, b, w, h = tunnel_bounds(self.pos_on_path, s_ref, w_nom=0.25, h_nom=0.25)   # w,h aktuell noch konstant
            self.tunnel_cache.append((c, n, b, w, h))
            whalf = 0.5 * w
            hhalf = 0.5 * h

            pos = obs["pos"]

            qc_val = MPCC_CFG["QC"]         # evtl. auf 5000+ erhöhen
            ql_val = MPCC_CFG["QL"]
            ramp   = min(1.0, (self._tick*self.dt) / MPCC_CFG["RAMP_TIME"])
            mu_val = MPCC_CFG["MU"] * ramp if j < self.N else 0.0

            # if straight-line tunnel is active, override pref to the tunnel center
            if self._use_line_tunnel:
                pref_i = c
            else:
                pref_i = pref

            p_vec = np.concatenate([
                pref_i,               # 0-2: reference point (path or tunnel center)
                tangent,              # 3-5
                [qc_val, ql_val, mu_val],   # 6-8
                n,                    # 9-11
                b,                    # 12-14
                [whalf, hhalf],       # 15-16
            ])

            self.acados_ocp_solver.set(j, "p", p_vec)
            if j == 0 and self._tick % 20 == 0:  # alle ~0.33 s
                # --- Parameter aus dem Solver zurücklesen
                p_dbg = self.acados_ocp_solver.get(0, "p")
                wh_dbg, hh_dbg = p_dbg[15], p_dbg[16]
                n_dbg, b_dbg   = p_dbg[9:12], p_dbg[12:15]

                # Abstand Drohne-Centerline
                e_n_dbg = np.dot(obs["pos"] - pref, n_dbg)
                e_b_dbg = np.dot(obs["pos"] - pref, b_dbg)

                # print(
                #     f"[t{self._tick:04d}] "
                #     f"s_ref={s_ref:5.2f}  "
                #     f"ec={np.linalg.norm(obs['pos']-pref):.3f}  "
                #     f"e_n={e_n_dbg:+.3f}/{wh_dbg:.2f}  "
                #     f"e_b={e_b_dbg:+.3f}/{hh_dbg:.2f}"
                # )

            delta = pos - pref
            cont_err = delta - (delta @ tangent) * tangent
            ec_norm = np.linalg.norm(cont_err)

            # --- debug: print all cost components for stage 0 -------------
            if j == 0:
                # errors
                lag_err = delta @ tangent
                el_sq_dbg = lag_err**2
                ec_sq_dbg = ec_norm**2

                # individual costs
                cost_ec = qc_val * ec_sq_dbg
                cost_el = ql_val * el_sq_dbg
                cost_mu = -mu_val * self.last_vtheta
                cost_df = 0.0
                cost_dv = 0.0
                gate_err_dbg = np.linalg.norm(pos - self._target_gate_pos) ** 2
                # cost_gate = q_gate * gate_err_dbg
                # omega cost uses current cmd values
                # omega_vec_dbg = self.last_rpy_cmd
                # cost_omega = MPCC_CFG["Q_OMEGA"] * np.dot(omega_vec_dbg, omega_vec_dbg)

                total_cost = cost_ec + cost_el + cost_mu 

                # print(
                #     f"[t={self._tick:04d}]  "
                #     f"ec²:{ec_sq_dbg:.4f}*{qc_val:.0f}={cost_ec:.3f},  "
                #     f"el²:{el_sq_dbg:.4f}*{ql_val:.0f}={cost_el:.3f},  "
                #     f"mu:{cost_mu:.3f},  "
                #     f"Δf:{cost_df:.3f},  Δv:{cost_dv:.3f},  "
                #     f"ω:{cost_omega:.3f},  Gate:{cost_gate:.3f}  "
                #     f"=> Total:{total_cost:.3f}"
                # )
        # build full 16‑element parameter vector for terminal stage (mu_val=0)
        # For terminal stage, use nominal gate penalty (no peak)
        p_terminal = np.concatenate([
            self.pos_on_path(theta_pred[-1]),
            tangent,
            [qc_val, ql_val, mu_val],  # qc, ql, mu
            n, b,
            [whalf, hhalf],
        ])

        self.acados_ocp_solver.set(self.N, "p", p_terminal)

        # ---------------- before solver.solve() -----------------
        # 1) copy the current state along the horizon
        for j in range(self.N + 1):
            self.acados_ocp_solver.set(j, "x", xcurrent)

        # 2) zero-order-hold for the previous inputs (or zeros)
        u_init = np.r_[self.last_f_cmd, self.last_rpy_cmd, 0.0]   # shape (nu,)
        for j in range(self.N):
            self.acados_ocp_solver.set(j, "u", u_init)

        # 3) enable full warm-start of multipliers
        self.acados_ocp_solver.options_set("qp_warm_start", 2)    # 2 = full


        status = self.acados_ocp_solver.solve()
        # im compute_control, nach dem Solve und dem Auslesen von x0_sol:

        if status != 0:                       # 0 = OK
            print(f"[acados] Abbruch mit Status {status}")
            return np.array([self.last_f_cmd, *self.last_rpy_cmd])

        lam = self.acados_ocp_solver.get(0, "lam")  # duals of dynamics
        # print(f"dual_norm={np.linalg.norm(lam):.1e}")
        qp_stat = self.acados_ocp_solver.get_stats("qp_stat")[-1]   # HPIPM exit flag
        qp_it   = self.acados_ocp_solver.get_stats("qp_iter")[-1]
        # print(f"QP flag={qp_stat}  qp_it={qp_it}")
        res = self.acados_ocp_solver.get_stats("residuals")
        max_res = np.max(res)
        # print(f"max_res={max_res:.1e}")
        # Check convergence
        if qp_it > 100 or max_res > 1e-4:
            # print("Warning: QP did not converge within 100 iterations or residual >1e-4")
            pass
        # 3b) min h_i der optimalen Stage-0-Lösung
        x0_sol = self.acados_ocp_solver.get(0, "x")
        p0_sol = self.acados_ocp_solver.get(0, "p")
        n_sol  = p0_sol[9:12];  b_sol  = p0_sol[12:15]
        wh_sol, hh_sol = p0_sol[15], p0_sol[16]
        pref_sol = p0_sol[0:3]
        d_sol = x0_sol[0:3] - pref_sol
        e_n = np.dot(d_sol, n_sol);  e_b = np.dot(d_sol, b_sol)
        h_vals = [wh_sol - e_n, wh_sol + e_n, hh_sol - e_b, hh_sol + e_b]
        # print(f"h_min={min(h_vals):+.3f}")



        # --- debug: print HPIPM residuals for clipping/numerical issues ------
        residuals = self.acados_ocp_solver.get_stats("residuals")
        max_res = np.max(residuals)
        # print(f"HPIPM max_residual: {max_res:.3e}")
        # (Predicted theta update removed: reference is now static)
        # retrieve solution afterwards
        x1 = self.acados_ocp_solver.get(1, "x")
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]
        # low-pass filter for theta and vtheta to prevent pref jumping ahead
        alpha = 1.0  # tuning: larger alpha => faster update
        self.last_theta = (1 - alpha) * self.last_theta + alpha * x1[-2]
        self.last_vtheta = (1 - alpha) * self.last_vtheta + alpha * x1[-1]

        cmd = x1[10:14]
        u_dbg = cmd
        # print(f"cmd f={u_dbg[0]:.2f}  rpy={u_dbg[1:]} ")

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
        self.tunnel_cache = [] 

    def get_trajectory(self) -> NDArray[np.floating]:
        """Get the trajectory points."""
        return self.traj_points

    def get_ref_point(self) -> NDArray[np.floating]:
        """Get the reference point."""
        return self._ref_point
    

def line_tunnel_bounds(p1: np.ndarray, p2: np.ndarray, s: float,
                       w_nom: float = 0.4, h_nom: float = 0.4):
    """Compute tunnel bounds along a straight line between p1 and p2.
    
    Args:
        p1, p2: Endpoints of the line segment (3D points).
        s: parameter in [0,1] indicating interpolation between p1 and p2.
        w_nom, h_nom: nominal width/height of the tunnel.

    Returns:
        c: center point on line at parameter s,
        n: lateral normal vector,
        b: vertical binormal vector,
        w_nom, h_nom.
    """
    # center point by linear interpolation
    c = (1 - s) * p1 + s * p2
    # direction from p1 to p2
    d = p2 - p1
    d = d / (np.linalg.norm(d) + 1e-9)
    # normal in horizontal plane
    n = np.array([-d[1], d[0], 0.0])
    n = n / (np.linalg.norm(n) + 1e-9)
    # binormal vertical
    b = np.cross(d, n)
    b = b / (np.linalg.norm(b) + 1e-9)
    return c, n, b, w_nom, h_nom

# Spline-based tunnel function that passes through multiple points
def spline_tunnel_bounds(points: np.ndarray, s: float,
                         w_nom: float = 0.4, h_nom: float = 0.4):
    """
    Compute tunnel bounds along a spline through given 3D points.
    Args:
        points: array of shape (M,3), control points defining the spline.
        s: parameter in [0,1] indicating interpolation along the spline.
        w_nom, h_nom: nominal width/height of the tunnel.
    Returns:
        c: center point on spline at parameter s,
        n: lateral normal vector,
        b: vertical binormal vector,
        w_nom, h_nom.
    """
    # extend points so tunnel is open at both ends
    first, second = points[0], points[1]
    last, before_last = points[-1], points[-2]
    p0 = 2 * first - second
    p_end = 2 * last - before_last
    ext_points = np.vstack([p0, points, p_end])
    # parameter grid for control points
    M = ext_points.shape[0]
    t_grid = np.linspace(0.0, 1.0, M)
    # build cubic splines for x,y,z
    cs_x = CubicSpline(t_grid, ext_points[:,0])
    cs_y = CubicSpline(t_grid, ext_points[:,1])
    cs_z = CubicSpline(t_grid, ext_points[:,2])
    # center point
    c = np.array([cs_x(s), cs_y(s), cs_z(s)])
    # tangent vector
    ds = 1e-3
    cp = np.array([cs_x(min(s+ds,1.0)), cs_y(min(s+ds,1.0)), cs_z(min(s+ds,1.0))])
    t = cp - c
    t = t / (np.linalg.norm(t) + 1e-9)
    # define normal via projection of global up-vector to avoid twisting
    up = np.array([0.0, 0.0, 1.0])
    n = up - np.dot(up, t) * t
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-6:
        # fallback normal if tangent aligned with up
        n = np.array([1.0, 0.0, 0.0])
    else:
        n = n / n_norm
    # binormal as cross-product to complete right-handed frame
    b = np.cross(t, n)
    b = b / (np.linalg.norm(b) + 1e-9)
    return c, n, b, w_nom, h_nom

def tunnel_bounds(pos_on_path, s: float,
                w_nom: float = 0.4, h_nom: float = 0.4):
    """Berechne Center-, Normal-, Binormal-Vektor & Breite/Höhe."""
    c      = pos_on_path(s)
    c_next = pos_on_path(s + 0.01)
    t      = c_next - c;  t /= np.linalg.norm(t) + 1e-9
    n      = np.array([-t[1], t[0], 0.0]); n /= np.linalg.norm(n) + 1e-9
    b      = np.cross(t, n);               b /= np.linalg.norm(b) + 1e-9
    return c, n, b, w_nom, h_nom

