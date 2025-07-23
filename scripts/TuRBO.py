"""TuRBO-based hyperparameter optimization for drone racing controller.

This module implements the TuRBO trust region Bayesian optimization algorithm
to tune controller hyperparameters using simulation-based evaluation.
"""
### As a short notice, this script is already runable, but still would need some refinement.
### We therefore decided not to include the necessary dependencies in the pyproject.toml file.
### Feel free to install torch gpytorch botorch manually, if interested in the script.

import logging
import math
from dataclasses import dataclass

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, standardize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from scripts.TuRBO_sim import simulate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("turbo.log", mode="a"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# --- Minimal TuRBO helper (aus Tutorial) -----------------------------------------
@dataclass
class TurboState:
    """Represents the state of the TuRBO trust region optimization.

    Attributes:
    ----------
    dim : int
        Dimensionality of the search space.
    batch_size : int
        Number of candidates evaluated per iteration.
    length : float
        Current trust region length.
    length_min : float
        Minimum allowed trust region length.
    length_max : float
        Maximum allowed trust region length.
    failure_counter : int
        Counter for consecutive failures.
    failure_tolerance : int
        Number of failures tolerated before shrinking the trust region.
    success_counter : int
        Counter for consecutive successes.
    success_tolerance : int
        Number of successes needed before expanding the trust region.
    best_value : float
        Best observed objective value.
    restart_triggered : bool
        Indicates if a restart is triggered due to trust region collapse.
    """

    dim: int
    batch_size: int
    length: float = 0.3
    length_min: float = 0.5 ** 5
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 0  # wird im __post_init__ gesetzt
    success_counter: int = 0
    success_tolerance: int = 4
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        """Initialize failure_tolerance based on dimension and batch size."""
        # Mindestens ceil(dim / batch_size) Failures erlauben
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )


def update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    """Update the TuRBO trust region state based on the latest batch of objective values.

    Parameters
    ----------
    state : TurboState
        The current state of the TuRBO optimizer.
    Y_next : torch.Tensor
        The batch of new objective values.

    Returns:
    -------
    TurboState
        The updated TuRBO state.
    """
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


def generate_batch(
    state: TurboState,
    model: SingleTaskGP,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int,
    n_candidates: int = 5000
) -> torch.Tensor:
    """Generate a batch of candidate points for TuRBO optimization using Max Posterior Sampling.

    Parameters
    ----------
    state : TurboState
        The current state of the TuRBO optimizer.
    model : SingleTaskGP
        The fitted Gaussian Process model.
    X : torch.Tensor
        The normalized input data.
    Y : torch.Tensor
        The standardized objective values.
    batch_size : int
        Number of candidates to generate.
    n_candidates : int, optional
        Number of candidate points to sample (default is 5000).

    Returns:
    -------
    torch.Tensor
        A tensor containing the batch of candidate points in normalized space.
    """
    dim, dtype, device = X.shape[-1], X.dtype, X.device
    # best observed center in normalized space
    x_center = X[Y.argmax(), :].clone()

    # gewichte Länge pro Achse
    ls = model.covar_module.lengthscale.squeeze().detach()
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
        4e-1,  # T_HORIZON
        50,    # BARRIER_WEIGHT (raised for new scale)
        0.45,   # TUNNEL_WIDTH
        0.2,   # NARROW_DIST
        0.05,  # GATE_FLAT_DIST
        1e-6,  # R_VTHETA
        -1e-2,  # REG_THRUST
        1e-4,  # REG_INPUTS
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
        1.2,   # T_HORIZON
        150,   # BARRIER_WEIGHT
        0.75,   # TUNNEL_WIDTH
        0.8,   # NARROW_DIST
        0.30,  # GATE_FLAT_DIST
        2.0,  # R_VTHETA
        0.2,   # REG_THRUST
        0.2,   # REG_INPUTS
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
    'QC','QL','MU','DVTHETA_MAX','T_HORIZON','BARRIER_WEIGHT','TUNNEL_WIDTH','NARROW_DIST',
    'GATE_FLAT_DIST','R_VTHETA','REG_THRUST','REG_INPUTS',
    'TUNNEL_WIDTH_GATE_0','TUNNEL_WIDTH_GATE_1','TUNNEL_WIDTH_GATE_2','TUNNEL_WIDTH_GATE_3'
]

start_param_dicts = [
    {
        'QC': 20,
        'QL': 20,
        'MU': 10,
        'DVTHETA_MAX': 1.7,
        'T_HORIZON': 0.9,
        'BARRIER_WEIGHT': 100,
        'TUNNEL_WIDTH': 0.6,
        'NARROW_DIST': 0.4,
        'GATE_FLAT_DIST': 0.15,
        'R_VTHETA': 0.008,
        'REG_THRUST': 0.08,
        'REG_INPUTS': 0.08, 
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


def evaluate_controller(params: torch.Tensor, n_eval: int = 1) -> float:
    """Evaluate the drone controller with the given hyperparameters.

    Parameters
    ----------
    params : torch.Tensor
        A tensor containing the hyperparameter values to evaluate.
    n_eval : int, optional
        Number of evaluation runs to perform (default is 1).

    Returns:
    -------
    float
        The reward value computed as the negative average time taken to finish the simulation runs.
    """
    hp = params.tolist()
    for _ in range(n_eval):
        param_dict = {
            'QC': hp[0],  'QL': hp[1],  'MU': hp[2],        'DVTHETA_MAX': hp[3],
            'T_HORIZON': hp[4], 'BARRIER_WEIGHT': hp[5],
            'TUNNEL_WIDTH': hp[6],     'NARROW_DIST': hp[7],
            'GATE_FLAT_DIST': hp[8],   'R_VTHETA': hp[9],
            'REG_THRUST': hp[10],       'REG_INPUTS': hp[11],
            'TUNNEL_WIDTH_GATE':   [hp[12], hp[13], hp[14], hp[15]],
        }
        # Debug: show which parameters are being evaluated
        print("PARAM_DICT:")
        for k, v in param_dict.items():
            print(f"  {k:18} = {v}")
        
        n_runs = 10
        time_finished, gates_passed, current_mass = simulate(
            n_runs=n_runs,
            gui=False,
            visualize=False,
            PARAM_DICT=param_dict
        )
        # Debug: print raw simulator outputs
        print(f'gates passed: {gates_passed}')
        print(f'time finished: {time_finished}')
        print(f'current mass: {current_mass}')
        valid_times = [t for t in time_finished if t is not None]
        if valid_times:
            avg_valid_time = sum(valid_times) / len(valid_times)
            print(f"Average Time: {avg_valid_time}")
        else:
            avg_valid_time = 1e10

        not_finished = 0
        best_times = []
        for i in time_finished:
            if i is not None:
                best_times.append(i)
            if i is None:
                not_finished += 1
                if not_finished > 5:
                    best_times.append(15)


        avg_time = sum(best_times) / len(best_times)
        reward = -avg_time



        avg_time = sum(best_times) / len(best_times)
        reward = -avg_time
        print(f"Reward: {reward}")
        finished = len(valid_times)/ n_runs
        logger.info(f"Average Time: {avg_valid_time}, Finisehd: {finished}, Time: {time_finished}, reward {reward}, used parameters: {param_dict}, current mass: {current_mass}")
    return reward

# --- Initial Design ------------------------------------------------------------
torch.manual_seed(0)

n_init = max(1, 0) 
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
train_Y_raw = Y_init.clone()
# -------------------------------------------------------------------------------

# --- TuRBO State & Loop --------------------------------------------------------
state = TurboState(dim=num_dims, batch_size=2)

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

    logger.info(f"Batch iteration {itr}: rewards {Y_next.view(-1).tolist()}, current TR length {state.length}")

    # TR-Status updaten
    state = update_state(state, Y_next)

    # neue Daten anhängen (RAW und anschließend neu standardisieren)
    train_X = torch.cat([train_X, X_next_norm], dim=0)
    train_Y_raw = torch.cat([train_Y_raw, Y_next], dim=0)
    train_Y = standardize(train_Y_raw)
    logger.info(f"Best raw reward so far: {train_Y_raw.max().item():.4f}")

    if state.restart_triggered:
        print(f"Trust region collapsed at iteration {itr}, restart triggered.")
        break

# bestes Ergebnis
best_idx = train_Y.argmax()
best_params = unnormalize(train_X[best_idx], bounds)
print("Best hyperparameters found:\n", best_params)