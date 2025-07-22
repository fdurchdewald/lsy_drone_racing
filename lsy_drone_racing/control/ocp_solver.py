import numpy as np
from casadi import MX, cos, sin, vertcat
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from lsy_drone_racing.control.mpccpp_config import MPCC_CFG

def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    model_name = "lsy_example_mpc"

    # Define Gravitational Acceleration
    GRAVITY = 9.806

    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    # Model Setting
    px = MX.sym("px")  # 0
    py = MX.sym("py")  # 1
    pz = MX.sym("pz")  # 2
    vx = MX.sym("vx")  # 3
    vy = MX.sym("vy")  # 4
    vz = MX.sym("vz")  # 5
    roll = MX.sym("r")  # 6
    pitch = MX.sym("p")  # 7
    yaw = MX.sym("y")  # 8
    f_collective = MX.sym("f_collective") # 9

    f_collective_cmd = MX.sym("f_collective_cmd") # 10
    r_cmd = MX.sym("r_cmd") # 11
    p_cmd = MX.sym("p_cmd") # 12
    y_cmd = MX.sym("y_cmd") # 13

    df_cmd = MX.sym("df_cmd") # 14
    dr_cmd = MX.sym("dr_cmd") # 15
    dp_cmd = MX.sym("dp_cmd") # 16
    dy_cmd = MX.sym("dy_cmd") # 17

    # MPCC extend
    theta = MX.sym("theta") # 18
    vtheta = MX.sym("vtheta") # 19
    dvtheta_cmd = MX.sym("dvtheta_cmd") # 20

    p = MX.sym("p", 18)  
    px_r, py_r, pz_r = p[0], p[1], p[2]
    tx, ty, tz = p[3], p[4], p[5]
    qc, ql, mu = p[6], p[7], p[8]
    nx, ny, nz = p[9], p[10], p[11]
    bx, by, bz = p[12], p[13], p[14]
    w_half = p[15]
    h_half = p[16]
    w_bar = p[17]


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
    droll = params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd
    dpitch = params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd
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

    # MPCC error terms
    pos = vertcat(px, py, pz)
    pref = vertcat(px_r, py_r, pz_r)
    t_vec = vertcat(tx, ty, tz)  
    delta = pos - pref

    lag_err = delta.T @ t_vec  
    cont_err = delta - lag_err * t_vec 

    ec_sq = cont_err.T @ cont_err  
    el_sq = lag_err**2  # e_l²

    # MPCC cost terms
    R_vth = MPCC_CFG["R_VTHETA"]

    n_vec = vertcat(nx, ny, nz)
    b_vec = vertcat(bx, by, bz)

    # Half-widths for the barrier function
    g_n_pos = n_vec.T @ delta - w_half  #  (p‑c)·n ≤  w/2
    g_n_neg = -n_vec.T @ delta - w_half
    g_b_pos = b_vec.T @ delta - h_half  #  (p‑c)·b ≤  h/2
    g_b_neg = -b_vec.T @ delta - h_half

    alpha = 50.0
    bar = (
        MX.log(1 + MX.exp(alpha * (g_n_pos)))
        + MX.log(1 + MX.exp(alpha * (g_n_neg)))
        + MX.log(1 + MX.exp(alpha * (g_b_pos)))
        + MX.log(1 + MX.exp(alpha * (g_b_neg)))
    )

    R_input = MPCC_CFG["REG_INPUTS"]
    R_thrust = MPCC_CFG["REG_THRUST"]

    # Stage cost
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
    model.cost_expr_ext_cost = stage_cost  
    model.p = p  
    model.con_h_expr = None 
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

    # Cost Type
    ocp.cost.cost_type = "EXTERNAL"

    # Tell acados how many parameters each stage has
    ocp.dims.np = 18 
    ocp.parameter_values = np.zeros(18)  

    # Define state constraints
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13, 15])
    lbx = np.array([0.1, 0.1, -1.3, -1.3, -1.3, -MPCC_CFG["DVTHETA_MAX"]])
    ubx = np.array([0.65, 0.65, 1.3, 1.3, 1.3, MPCC_CFG["DVTHETA_MAX"]])
    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx

    # Define input constraints
    ocp.constraints.nh = 0  # Dimension 0
    ocp.constraints.expr_h = None  # nichts mehr verknüpft
    ocp.constraints.lh = np.zeros((0,))  # leere NumPy-Arrays
    ocp.constraints.uh = np.zeros((0,))

    dvtheta_max = MPCC_CFG["DVTHETA_MAX"]  
    ocp.constraints.lbu = np.array([-dvtheta_max])
    ocp.constraints.ubu = np.array([dvtheta_max])
    ocp.constraints.idxbu = np.array([4])  

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

    ocp.solver_options.nlp_solver_max_iter = 300
    ocp.solver_options.qp_solver_warm_start = 0
    ocp.solver_options.qp_solver_iter_max = 300

    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.globalization_line_search_use_sufficient_descent = 1

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc.json", verbose=verbose)

    return acados_ocp_solver, ocp