import math
from dataclasses import dataclass

import torch
from scripts.TuRBO_sim import simulate
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize, standardize
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
from lsy_drone_racing.control.debug_utils import get_logger as logger

log = logger('TuRBO')
# --- Minimal TuRBO helper (aus Tutorial) -----------------------------------------
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.3
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 0  # wird im __post_init__ gesetzt
    success_counter: int = 0
    success_tolerance: int = 5
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        # Mindestens ceil(dim / batch_size) Failures erlauben
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )


def update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    if Y_next.max() > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.failure_counter += 1
        state.success_counter = 0

    if state.success_counter >= state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, Y_next.max().item())
    if state.length < state.length_min:
        state.restart_triggered = True

    return state


def generate_batch(state: TurboState, model, X, Y, batch_size: int, n_candidates: int = 5000):
    dim, dtype, device = X.shape[-1], X.dtype, X.device
    # best observed center in normalized space
    x_center = X[Y.argmax(), :].clone()

    # gewichte Länge pro Achse
    ls = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = ls / ls.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    sobol = SobolEngine(dim, scramble=True)
    X_cand = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    X_cand = tr_lb + (tr_ub - tr_lb) * X_cand

    # Max Posterior Sampling für Batch-Kandidaten
    from botorch.generation import MaxPosteriorSampling
    ts = MaxPosteriorSampling(model=model, replacement=False)
    return ts(X_cand, num_samples=batch_size)
# -------------------------------------------------------------------------------

# --- Hyperparameter-Bounds (24 Parameter) --------------------------------------
# Order MUST match hyperparam_names below. The last 8 dims are 4 OBSTACLE_RADIUS
# values and 4 per-gate TUNNEL_WIDTH_GATE values.
bounds = torch.tensor([
    [   # lower bounds
        5,     # QC
        5,     # QL
        1e-4,  # MU
        1.0,   # DVTHETA_MAX
        10,    # N
        1e-3,  # T_HORIZON
        0.0,   # ALPHA_INTERP
        1e-5,  # BW_RAMP
        0.3,  # RAMP_TIME
        10,    # BARRIER_WEIGHT (raised for new scale)
        0.4,   # TUNNEL_WIDTH
        0.2,   # NARROW_DIST
        0.05,  # GATE_FLAT_DIST
        1e-5,  # R_VTHETA
        1e-4,  # REG_THRUST
        1e-4,  # REG_INPUTS
        0.08,  # OBSTACLE_RADIUS_0
        0.08,  # OBSTACLE_RADIUS_1
        0.08,  # OBSTACLE_RADIUS_2
        0.08,  # OBSTACLE_RADIUS_3
        0.10,  # TUNNEL_WIDTH_GATE_0
        0.10,  # TUNNEL_WIDTH_GATE_1
        0.10,  # TUNNEL_WIDTH_GATE_2
        0.10,  # TUNNEL_WIDTH_GATE_3
    ],
    [   # upper bounds
        60,    # QC
        60,    # QL
        100,    # MU
        3.0,   # DVTHETA_MAX
        30,    # N
        1.0,   # T_HORIZON
        1.0,   # ALPHA_INTERP
        0.5,   # BW_RAMP
        1.3,   # RAMP_TIME
        150,   # BARRIER_WEIGHT
        0.8,   # TUNNEL_WIDTH
        0.8,   # NARROW_DIST
        0.30,  # GATE_FLAT_DIST
        0.05,  # R_VTHETA
        0.2,   # REG_THRUST
        0.2,   # REG_INPUTS
        0.20,  # OBSTACLE_RADIUS_0
        0.20,  # OBSTACLE_RADIUS_1
        0.20,  # OBSTACLE_RADIUS_2
        0.20,  # OBSTACLE_RADIUS_3
        0.40,  # TUNNEL_WIDTH_GATE_0
        0.40,  # TUNNEL_WIDTH_GATE_1
        0.40,  # TUNNEL_WIDTH_GATE_2
        0.40,  # TUNNEL_WIDTH_GATE_3
    ]
], dtype=torch.double)
num_dims = bounds.shape[1]
# -------------------------------------------------------------------------------

# --- Custom Start-Param-Dicts --------------------------------------------
# Definiere Start-Konfigurationen als Liste von Dictionaries:
hyperparam_names = [
    'QC','QL','MU','DVTHETA_MAX','N','T_HORIZON','ALPHA_INTERP',
    'BW_RAMP','RAMP_TIME','BARRIER_WEIGHT','TUNNEL_WIDTH','NARROW_DIST',
    'GATE_FLAT_DIST','R_VTHETA','REG_THRUST','REG_INPUTS',
    'OBSTACLE_RADIUS_0','OBSTACLE_RADIUS_1','OBSTACLE_RADIUS_2','OBSTACLE_RADIUS_3',
    'TUNNEL_WIDTH_GATE_0','TUNNEL_WIDTH_GATE_1','TUNNEL_WIDTH_GATE_2','TUNNEL_WIDTH_GATE_3'
]

start_param_dicts = [
    {
        'QC': 20,
        'QL': 20,
        'MU': 10,
        'DVTHETA_MAX': 1.9,
        'N': 20,
        'T_HORIZON': 0.9,
        'ALPHA_INTERP': 0.6,
        'BW_RAMP': 0.2,
        'RAMP_TIME': 0.5,
        'BARRIER_WEIGHT': 100,
        'TUNNEL_WIDTH': 0.6,
        'NARROW_DIST': 0.4,
        'GATE_FLAT_DIST': 0.15,
        'R_VTHETA': 0.008,
        'REG_THRUST': 0.08,
        'REG_INPUTS': 0.08,
        'OBSTACLE_RADIUS_0': 0.11,
        'OBSTACLE_RADIUS_1': 0.14,
        'OBSTACLE_RADIUS_2': 0.10,
        'OBSTACLE_RADIUS_3': 0.10,
        'TUNNEL_WIDTH_GATE_0': 0.25,
        'TUNNEL_WIDTH_GATE_1': 0.15,
        'TUNNEL_WIDTH_GATE_2': 0.18,
        'TUNNEL_WIDTH_GATE_3': 0.25,
    },
]

# Umwandeln in Tensor (n_start × num_dims)
start_points = torch.tensor(
    [[d[name] for name in hyperparam_names] for d in start_param_dicts],
    dtype=torch.double
)
# ---------------------------------------------------------------------------


def evaluate_controller(params: torch.Tensor, n_eval: int = 3) -> float:
    """
    Ruft simulate() mit allen Hyperparametern auf.
    Reward:
      - alle 4 Gates geschafft: reward = -time_finished
      - Gate(s) verfehlt:      reward = -1e6 - 1e5*(fehlende Gates)
    """
    hp = params.tolist()
    for _ in range(n_eval):
        param_dict = {
            'QC': hp[0],
            'QL': hp[1],
            'MU': hp[2],
            'DVTHETA_MAX': hp[3],
            'N': int(round(hp[4])),
            'T_HORIZON': hp[5],
            'ALPHA_INTERP': hp[6],
            'BW_RAMP': hp[7],
            'RAMP_TIME': hp[8],
            'BARRIER_WEIGHT': hp[9],
            'TUNNEL_WIDTH': hp[10],
            'NARROW_DIST': hp[11],
            'GATE_FLAT_DIST': hp[12],
            'R_VTHETA': hp[13],
            'REG_THRUST': hp[14],
            'REG_INPUTS': hp[15],
            'OBSTACLE_RADIUS': [hp[16], hp[17], hp[18], hp[19]],
            'TUNNEL_WIDTH_GATE': [hp[20], hp[21], hp[22], hp[23]],
        }
        # Debug: show which parameters are being evaluated
        print("PARAM_DICT:")
        for k, v in param_dict.items():
            print(f"  {k:18} = {v}")
        time_finished, gates_passed, dist_z = simulate(
            n_runs=10,
            gui=False,
            visualize=False,
            PARAM_DICT=param_dict
        )
        # Debug: print raw simulator outputs
        print(f'gates passed: {gates_passed}')
        print(f'time finished: {time_finished}')
        valid_times = [t for t in time_finished if t is not None]
        if valid_times:
            avg_valid_time = sum(valid_times) / len(valid_times)
            print(f"Average Time: {avg_valid_time}")


        count_notfinished = 0
        total_finished = []
        for t in time_finished:
            if t is not None:
                total_finished.append(t)
            else:
                count_notfinished += 1
                if count_notfinished > 2:
                    total_finished.append(30.0)
        avg_time = sum(total_finished) / len(total_finished)
        reward = -avg_time
        print(f"Reward: {reward}")
        log.debug(f"Time: {time_finished}, reward {reward}, used parameters: {param_dict}, dist z {dist_z}")
    return reward

# --- Initial Design ------------------------------------------------------------
torch.manual_seed(0)

n_init = max(2, 4 * num_dims)
num_start = start_points.size(0)
num_rest = n_init - num_start
if num_rest > 0:
    # Sobol samples come back as (num_rest, 1, num_dims), reshape to (num_rest, num_dims)
    rest = draw_sobol_samples(bounds=bounds, n=num_rest, q=1, seed=0).reshape(num_rest, num_dims)
    X_init = torch.cat([start_points, rest], dim=0)
else:
    X_init = start_points
Y_init = torch.tensor([evaluate_controller(x) for x in X_init],
                      dtype=torch.double).unsqueeze(-1)

train_X = normalize(X_init, bounds)
train_Y = standardize(Y_init)
# -------------------------------------------------------------------------------

# --- TuRBO State & Loop --------------------------------------------------------
state = TurboState(dim=num_dims, batch_size=4)

n_iter = 1000
for itr in range(n_iter):
    # GP anpassen
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Generiere eine Batch von Kandidaten entsprechend state.batch_size
    X_next_norm = generate_batch(state, gp, train_X, train_Y, batch_size=state.batch_size)

    # Unnormalize alle Kandidaten (Form: (batch_size, dim))
    X_next = unnormalize(X_next_norm, bounds)

    # Jede Kandidaten-Konfiguration evaluieren
    Y_list = []
    for cand in X_next:
        y_val = evaluate_controller(cand)
        Y_list.append([y_val])
    Y_next = torch.tensor(Y_list, dtype=torch.double)

    log.debug(f"Batch iteration {itr}: rewards {Y_next.view(-1).tolist()}, current TR length {state.length}")

    # TR-Status updaten
    state = update_state(state, Y_next)

    # neue Daten anhängen
    train_X = torch.cat([train_X, X_next_norm], dim=0)
    train_Y = torch.cat([
        train_Y,
        standardize(Y_next, train_Y.mean(), train_Y.std())
    ], dim=0)

    if state.restart_triggered:
        print(f"Trust region collapsed at iteration {itr}, restart triggered.")
        break

# bestes Ergebnis
best_idx = train_Y.argmax()
best_params = unnormalize(train_X[best_idx], bounds)
print("Best hyperparameters found:\n", best_params)