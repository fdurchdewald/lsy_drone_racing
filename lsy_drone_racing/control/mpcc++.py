"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
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
    MU=0.05,
    DVTHETA_MAX=1,
    N=5,
    T_HORIZON=0.5,
    RAMP_TIME=2.0,
    BARRIER_WEIGHT = 100, 
    TUNNEL_WIDTH = 0.25,  # nominal tunnel width
    TUNNEL_SLACK_LINEAR = 0.0,
    TUNNEL_SLACK_QUAD = 0.0
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

    #MPCC++
    tunnel_n = MX.sym("e_n") 
    tunnel_b = MX.sym("e_b")
    # parameter vector: [pref(3), t(3), qc, ql, mu, pg(3), q_gate,   R_DF, R_DV, Q_OMEGA]
    # 
    p = MX.sym("p", 17)
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
        tunnel_n,
        tunnel_b, 
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
        MX.zeros(2,1)
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
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs
    model.cost_expr_ext_cost = stage_cost  # stage cost
    model.cost_expr_ext_cost_e = term_cost  # terminal cost
    model.p = p  # expose parameters to the OCP
    return model


def create_ocp_solver(
    Tf: float, N: int, verbose: bool = True
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
    ocp.cost.cost_type_e = "EXTERNAL"

    # Tell acados how many parameters each stage has
    ocp.dims.np = 17
    # default parameter vector (will be overwritten at run time)
    ocp.parameter_values = np.zeros(17)

    # Set state constraints: collective thrust, Euler commands, and virtual progress speed
# statt der 17-Einträge-Arrays:
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13, 15, 16, 17])

    # Jetzt nur noch 8 Werte pro Grenze:
    ocp.constraints.lbx = np.array([
        0.1,    # f_collective (x[ 9 ])
        0.1,    # f_collective_cmd (x[10])
    -1.57,  # r_cmd (x[11])
    -1.57,  # p_cmd (x[12])
    -1.57,  # y_cmd (x[13])
        0.0,   # vtheta (x[15])
    -MPCC_CFG["TUNNEL_WIDTH"]/2,  # tunnel_n (x[16])
    -MPCC_CFG["TUNNEL_WIDTH"]/2   # tunnel_b (x[17])
    ])
    ocp.constraints.ubx = np.array([
        0.55,   # f_collective
        0.55,   # f_collective_cmd
        1.0,    # r_cmd
        1.0,    # p_cmd
        1.0,    # y_cmd
        MPCC_CFG["DVTHETA_MAX"],  # vtheta
        MPCC_CFG["TUNNEL_WIDTH"]/2,  # tunnel_n
        MPCC_CFG["TUNNEL_WIDTH"]/2   # tunnel_b
    ])
    # Und die Slack-Indizes beziehen sich nun auf die Positionen 6 und 7 in diesem 8-Element-Array:
    ocp.constraints.idxsbx = np.array([6, 7])

    zl = MPCC_CFG["TUNNEL_SLACK_LINEAR"] * np.ones(2)
    zu = MPCC_CFG["TUNNEL_SLACK_LINEAR"] * np.ones(2)
    Zl = MPCC_CFG["TUNNEL_SLACK_QUAD"]   * np.ones(2)
    Zu = MPCC_CFG["TUNNEL_SLACK_QUAD"]   * np.ones(2)

    ocp.cost.zl = zl
    ocp.cost.zu = zu
    ocp.cost.Zl = Zl
    ocp.cost.Zu = Zu

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
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # GAUSS_NEWTON
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.tol = 1e-7        
    ocp.solver_options.tf = 1.0


    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.nlp_solver_ext_qp_res = 1

    ocp.solver_options.nlp_solver_max_iter = 300
    ocp.solver_options.qp_solver_warm_start = 0
    ocp.solver_options.qp_solver_iter_max = 10000

    ocp.solver_options.regularize_method  = "CONVEXIFY"
    ocp.solver_options.line_search_use_sufficient_descent = 1

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc.json", verbose=verbose)
    acados_integrator = AcadosSimSolver(ocp, json_file = "lsy_example_mpc.json", verbose=verbose)


    return acados_ocp_solver, acados_integrator, ocp


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
        vis_s = np.linspace(0.0, self.s_total, 200, endpoint=False)
        self.traj_points = np.column_stack((cs_x(vis_s), cs_y(vis_s), cs_z(vis_s)))

        # ------------------------------------------------------------------
        # MPC settings (unchanged)
        # ------------------------------------------------------------------
        self.N = MPCC_CFG["N"]  # number of steps in the MPC
        self.T_HORIZON = MPCC_CFG["T_HORIZON"]  # time span of the MPC
        self.dt = self.T_HORIZON / self.N

        self.acados_ocp_solver, self.acados_integrator, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)
    
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
        self._tunnel_n = 0.0
        self._tunnel_b = 0.0

        # Store predicted theta trajectory for reference in next iteration
        self._predicted_theta = np.zeros(self.N + 1)
        def _pos_on_path(s: float) -> np.ndarray:
            s_wrap = s % self.s_total
            return np.array([self._cs_x(s_wrap),
                            self._cs_y(s_wrap),
                            self._cs_z(s_wrap)])
        self.pos_on_path = _pos_on_path          # <-- expose helper
        # Precompute tunnel geometry at high resolution along full path
        self.vis_s = vis_s  # array of 200 sample arclengths
        self.tunnel_cache = []
        w_nom = MPCC_CFG["TUNNEL_WIDTH"]
        h_nom =MPCC_CFG["TUNNEL_WIDTH"]
        for s in vis_s:
            c, n, b, w, h = tunnel_bounds(self.pos_on_path, s, w_nom=w_nom, h_nom=h_nom)
            self.tunnel_cache.append((c, n, b, w, h))
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
                np.zeros(2),  # 2: tunnel_n, tunnel_b (slack variables)
            )
        )

        try:
            vtheta_pred = np.zeros(self.N)
            for j in range(1, self.N+1):
                xj = self.acados_ocp_solver.get(j, "x")
                vtheta_pred[j-1] = float(xj[15])
            theta_pred = np.zeros(self.N + 1)
            theta_pred[0] = self.last_theta
            for j in range(self.N):
                theta_pred[j+1] = theta_pred[j] + vtheta_pred[j] * self.dt
        except Exception:
            # Erster Aufruf: noch keine vorherigen Prädiktionen
            theta_pred = self.last_theta + self.last_vtheta * np.linspace(0.0, self.T_HORIZON, self.N + 1)

        if self._tick % 20 == 0:
            print(f"θ={self.last_theta:6.2f}  vθ={self.last_vtheta:4.2f}")


        # upon gate change, generate full straight tunnel once
        planned_traj = np.zeros((self.N + 1, 3))
        for j in range(self.N + 1):
            xj = self.acados_ocp_solver.get(j, "x")
            planned_traj[j] = xj[:3]   # first three state entries: px, py, pz
        self._planned_traj = planned_traj

        for j in range(self.N):
            s_ref = theta_pred[j]
            pref = self.pos_on_path(s_ref)
            self._ref_point = pref  
            pref_next = self.pos_on_path(s_ref + 0.001)
            tangent = unit(pref_next - pref)

            idx_vis = min(range(len(self.vis_s)), key=lambda k: abs(self.vis_s[k] - s_ref))
            c, n, b, w, h = self.tunnel_cache[idx_vis]
            whalf = w / 2.0
            hhalf = h / 2.0
            pos = obs["pos"]

            tunnel_e_val = np.dot(pos - c, n)
            tunnel_b_val = np.dot(pos - c, b)

            xcurrent[-2] = tunnel_e_val
            xcurrent[-1] = tunnel_b_val
            
            self.acados_ocp_solver.set(0, "lbx", xcurrent)
            self.acados_ocp_solver.set(0, "ubx", xcurrent)

            qc_val = MPCC_CFG["QC"]         
            ql_val = MPCC_CFG["QL"]
            ramp   = min(1.0, (self._tick*self.dt) / MPCC_CFG["RAMP_TIME"])
            mu_val = MPCC_CFG["MU"] * ramp if j < self.N else 0.0


            p_vec = np.concatenate([
                pref,               # 0-2: reference point (path or tunnel center)
                tangent,              # 3-5
                [qc_val, ql_val, mu_val],   # 6-8
                n,                    # 9-11
                b,                    # 12-14
                [whalf, hhalf],       # 15-16
            ])
            self.acados_ocp_solver.set(j, "p", p_vec)
            self.acados_integrator.set("x", xcurrent)
            ucurrent = self.acados_ocp_solver.get(j, "u")  # shape (nu,)
            self.acados_integrator.set("u", ucurrent)
            status = self.acados_integrator.solve()

        p_terminal = np.concatenate([
            self.pos_on_path(theta_pred[-1]),
            tangent,
            [qc_val, ql_val, mu_val],  # qc, ql, mu
            n, b,
            [whalf, hhalf],
        ])

        self.acados_ocp_solver.set(self.N, "p", p_terminal)

        # ---------------- before solver.solve() -----------------

        u_init = np.r_[self.last_f_cmd, self.last_rpy_cmd, 0.0]   # shape (nu,)
        for j in range(self.N):
            self.acados_ocp_solver.set(j, "u", u_init)

        self.acados_ocp_solver.options_set("qp_warm_start", 2)    # 2 = full



        # 2) Override tunnel box constraints for stages 1..N-1 (already set inside loop),
        #    now also set for terminal stage N:

        # 3) Finally solve
        status = self.acados_ocp_solver.solve()
        qp_it = self.acados_ocp_solver.get_stats("qp_iter")[-1]
        max_res = np.max(self.acados_ocp_solver.get_stats("residuals"))

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

        cmd = x1[10:14]
        self.last_theta = x1[14]
        self.last_vtheta = x1[15]
        if x1[16] < -MPCC_CFG["TUNNEL_WIDTH"]/2 or x1[17] < -MPCC_CFG["TUNNEL_WIDTH"]/2 or x1[16] > MPCC_CFG["TUNNEL_WIDTH"]/2 or x1[17] > MPCC_CFG["TUNNEL_WIDTH"]/2:
            print(f"Warning: tunnel_n/b out of bounds: {x1[16]:.3f}, {x1[17]:.3f}")
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
    


def tunnel_bounds(pos_on_path, s: float,
                w_nom: float = 0.4, h_nom: float = 0.4):
    """Berechne Center-, Normal-, Binormal-Vektor & Breite/Höhe."""
    c      = pos_on_path(s)
    c_next = pos_on_path(s + 0.01)
    t      = c_next - c;  t /= np.linalg.norm(t) + 1e-9
    n      = np.array([-t[1], t[0], 0.0]); n /= np.linalg.norm(n) + 1e-9
    b      = np.cross(t, n);               b /= np.linalg.norm(b) + 1e-9
    return c, n, b, w_nom, h_nom

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

# helper: position on the reference path for a given arclength s
def pos_on_path(s: float) -> np.ndarray:
    s_wrap = s % self.s_total
    return np.array([self._cs_x(s_wrap), self._cs_y(s_wrap), self._cs_z(s_wrap)])
